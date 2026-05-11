"""
Decoupled Flow-Matching + Stable Linear SDE training on LASA 2D.

Single-script "hybrid" runner that performs Phase A (Conditional Flow Matching
+ temporal regularization), Phase B (latent inversion of all expert points),
and Phase C (stable linear SDE regression) inside one outer loop. Checkpoints
and LASA visualizers are saved every ``--save-every`` macro-epochs (and on the
last macro-epoch).
"""
import argparse
import os

import numpy as np
import torch
import torch.optim as optim

from iflow import model
from iflow.dataset import lasa_dataset
from iflow.dataset.lasa_spatial_dataset import build_lasa_samplers
from iflow.trainers.fm_latent_sde_train import (
    build_z_pairs_from_trajectories,
    loss_phase_c,
    phase_a_step,
)
from iflow.utils import makedirs, to_torch
from iflow.visualization import (
    visualize_2d_generated_trj,
    visualize_latent_distribution,
    visualize_vector_field,
)


def parse_args():
    p = argparse.ArgumentParser(description="Decoupled FM + stable SDE on LASA")
    p.add_argument("--shape", type=str, default="Bump")
    p.add_argument("--device", type=str, default="cpu", choices=("cpu", "cuda"))
    p.add_argument("--macro-epochs", type=int, default=400)
    p.add_argument("--phase-a-iters", type=int, default=400)
    p.add_argument("--phase-c-iters", type=int, default=200)
    p.add_argument("--fm-batch", type=int, default=512)
    p.add_argument("--reg-batch", type=int, default=16)
    p.add_argument("--reg-window", type=int, default=8)
    p.add_argument("--reg-every-k", type=int, default=10)
    p.add_argument("--lambda-reg", type=float, default=0.1)
    p.add_argument("--lambda-warmup", type=int, default=2,
                   help="Macro-epochs with lambda_reg=0 before ramping in")
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--lr-spatial", type=float, default=1e-3)
    p.add_argument("--lr-sde", type=float, default=1e-2)
    p.add_argument("--hidden", type=str, default="64-64-64")
    p.add_argument("--encode-chunk", type=int, default=2048)
    p.add_argument("--out-name", type=str, default="lasa_fm_hybrid")
    p.add_argument(
        "--save-every",
        type=int,
        default=25,
        help="Save checkpoints and plots every N macro-epochs (and always on the last)",
    )
    return p.parse_args()


def build_models(dim, hidden_dims, dt, device):
    velocity_net = model.SpatialVelocityNet(dim=dim, hidden_dims=hidden_dims, layer_type="concat", nonlinearity="tanh")
    spatial_ode = model.NeuralODEFlow(velocity_net=velocity_net, train_solver="euler", train_solver_steps=20)
    dummy = model.DummyLinearPredictor(dim=dim, init_diag=-1.0)
    sde = model.StableLinearSDE(dim=dim, dt=dt)

    spatial_ode.to(device)
    dummy.to(device)
    sde.to(device)
    return spatial_ode, dummy, sde


def main():
    args = parse_args()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable; using CPU.")

    data = lasa_dataset.LASA(filename=args.shape, device=device)
    dim = data.dim
    fm_sampler, reg_sampler = build_lasa_samplers(data, device, window_length=args.reg_window)

    hidden_dims = tuple(int(h) for h in args.hidden.split("-"))
    spatial_ode, dummy, sde = build_models(dim, hidden_dims, args.dt, device)

    composite = model.DecoupledFMImitationFlow(
        spatial_ode=spatial_ode, dynamics=sde, dim=dim, device=device
    ).to(device)

    optim_a = optim.Adam(
        list(spatial_ode.parameters()) + list(dummy.parameters()), lr=args.lr_spatial
    )
    optim_c = optim.Adam(sde.parameters(), lr=args.lr_sde)

    plot_dir = os.path.join(
        os.path.dirname(__file__), "lasa_fm_plots", args.shape, args.out_name
    )
    weights_dir = os.path.join(os.path.dirname(__file__), "lasa_fm_weights", args.shape)
    makedirs(plot_dir)
    makedirs(weights_dir)

    global_step = 0
    for macro_epoch in range(args.macro_epochs):
        if macro_epoch < args.lambda_warmup:
            current_lambda = 0.0
        else:
            ramp = min(1.0, (macro_epoch - args.lambda_warmup + 1) / max(args.lambda_warmup, 1))
            current_lambda = args.lambda_reg * ramp

        spatial_ode.train()
        dummy.train()

        running = {"loss": 0.0, "loss_fm": 0.0, "loss_reg": 0.0, "n_reg": 0}
        for it in range(args.phase_a_iters):
            stats = phase_a_step(
                spatial_ode=spatial_ode,
                dummy_predictor=dummy,
                optimizer=optim_a,
                fm_sampler=fm_sampler,
                reg_sampler=reg_sampler,
                fm_batch_size=args.fm_batch,
                reg_batch_size=args.reg_batch,
                dt=args.dt,
                lambda_reg=current_lambda,
                step_idx=it,
                reg_every_k=args.reg_every_k,
            )
            running["loss"] += stats["loss"]
            running["loss_fm"] += stats["loss_fm"]
            if stats["loss_reg"] is not None:
                running["loss_reg"] += stats["loss_reg"]
                running["n_reg"] += 1
            global_step += 1

        avg_loss = running["loss"] / max(args.phase_a_iters, 1)
        avg_fm = running["loss_fm"] / max(args.phase_a_iters, 1)
        avg_reg = running["loss_reg"] / running["n_reg"] if running["n_reg"] > 0 else float("nan")
        print(
            "[macro {:03d}] phase A | lambda={:.3g} loss={:.4f} fm={:.4f} reg={:.4f} (n_reg={})".format(
                macro_epoch, current_lambda, avg_loss, avg_fm, avg_reg, running["n_reg"]
            )
        )

        spatial_ode.eval()
        with torch.no_grad():
            trajectories_z = []
            for tr in data.train_data:
                y_t = torch.from_numpy(np.asarray(tr)).float().to(device)
                z_t = spatial_ode.encode_batched(y_t, chunk_size=args.encode_chunk, training=False)
                trajectories_z.append(z_t)
        z_pairs = build_z_pairs_from_trajectories(trajectories_z).detach()

        sde.train()
        for it in range(args.phase_c_iters):
            optim_c.zero_grad()
            l_c = loss_phase_c(sde, z_pairs, dt=args.dt)
            l_c.backward()
            optim_c.step()
        eigvals = sde.eigvals()
        print(
            "[macro {:03d}] phase C | mse={:.6f} stable={} max_re={:.4f}".format(
                macro_epoch,
                float(l_c.detach()),
                bool((eigvals.real < 0).all()),
                float(eigvals.real.max()),
            )
        )

        save_outputs = (macro_epoch % args.save_every == 0) or (
            macro_epoch == args.macro_epochs - 1
        )
        if save_outputs:
            with torch.no_grad():
                composite.eval()
                visualize_2d_generated_trj(
                    data.train_data, composite, device, fig_number=2,
                    save_path=os.path.join(plot_dir, "macro_{:04d}_traj2d.png".format(macro_epoch)),
                )
                visualize_latent_distribution(
                    data.train_data, composite, device, fig_number=1,
                    save_path=os.path.join(plot_dir, "macro_{:04d}_latent.png".format(macro_epoch)),
                )
                visualize_vector_field(
                    data.train_data, composite, device, fig_number=3,
                    save_path=os.path.join(plot_dir, "macro_{:04d}_vector_field.png".format(macro_epoch)),
                )

            ckpt_path = os.path.join(weights_dir, "{}_macro{:04d}.pt".format(args.out_name, macro_epoch))
            torch.save(
                {
                    "shape": args.shape,
                    "macro_epoch": macro_epoch,
                    "dim": dim,
                    "hidden_dims": list(hidden_dims),
                    "dt": args.dt,
                    "spatial_ode_state_dict": spatial_ode.state_dict(),
                    "dummy_state_dict": dummy.state_dict(),
                    "sde_state_dict": sde.state_dict(),
                    "lasa_mean": data.mean,
                    "lasa_std": data.std,
                },
                ckpt_path,
            )
            print("Saved checkpoint: {}".format(ckpt_path))


if __name__ == "__main__":
    main()
