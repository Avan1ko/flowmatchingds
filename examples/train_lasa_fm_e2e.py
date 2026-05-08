"""
End-to-end Flow Matching + stability-preserving diffeomorphism on LASA 2D.

Single training script that learns a Neural-ODE diffeomorphism (trained with
simulation-free Conditional Flow Matching) and a provably-Hurwitz linear latent
SDE *jointly*, under one combined loss and one optimizer. No Phase A/B/C
alternation as in ``train_lasa_fm_hybrid.py``.

Per step, the optimizer minimizes

    L_total = L_FM(spatial_ode) + alpha * L_SDE_pairs(spatial_ode, sde)

where:
    - L_FM is the simulation-free CFM loss matching the OT straight-line
      target between a Gaussian base and the union of expert points.
    - L_SDE_pairs encodes consecutive demo pairs ``(y_t, y_{t+1})`` to
      ``(z_t, z_{t+1})`` *through the inverse ODE* (gradients flow into the
      diffeomorphism) and regresses the stable linear SDE drift
      ``A = S - D`` (skew-symmetric S minus SPD D, hence Hurwitz).

Stability: a diffeomorphism conjugates dynamics, so global asymptotic
stability of ``dz = A z dt + K dB`` at the origin transfers to global
asymptotic stability of ``y = phi(z)`` at ``phi(0)``.
"""
import argparse
import os

import torch
import torch.optim as optim

from iflow import model
from iflow.dataset import lasa_dataset
from iflow.dataset.lasa_spatial_dataset import build_lasa_samplers
from iflow.trainers.fm_latent_sde_train import (
    encode_window,
    loss_fm,
    loss_phase_c,
)
from iflow.utils import makedirs
from iflow.visualization import (
    visualize_2d_generated_trj,
    visualize_latent_distribution,
    visualize_vector_field,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="End-to-end FM + stable-SDE imitation flow on LASA"
    )
    p.add_argument("--shape", type=str, default="Angle")
    p.add_argument("--device", type=str, default="cpu", choices=("cpu", "cuda"))
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--steps-per-epoch", type=int, default=200)
    p.add_argument("--fm-batch", type=int, default=512)
    p.add_argument("--pair-batch", type=int, default=64)
    p.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Target weight on the latent SDE regression loss",
    )
    p.add_argument(
        "--alpha-warmup",
        type=int,
        default=2,
        help="Epochs at alpha=0 before linearly ramping in over the same span",
    )
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=str, default="64-64-64")
    p.add_argument("--encode-chunk", type=int, default=2048)
    p.add_argument("--vis-every", type=int, default=25)
    p.add_argument("--ckpt-every", type=int, default=25)
    p.add_argument("--out-name", type=str, default="lasa_fm_e2e")
    return p.parse_args()


def build_models(dim, hidden_dims, dt, device):
    velocity_net = model.SpatialVelocityNet(
        dim=dim, hidden_dims=hidden_dims, layer_type="concat", nonlinearity="tanh"
    )
    spatial_ode = model.NeuralODEFlow(
        velocity_net=velocity_net, train_solver="euler", train_solver_steps=20
    )
    sde = model.StableLinearSDE(dim=dim, dt=dt)

    spatial_ode.to(device)
    sde.to(device)
    return spatial_ode, sde


def alpha_schedule(epoch, target_alpha, warmup_epochs):
    """Linear warmup: alpha=0 for the first ``warmup_epochs``, then ramps to
    ``target_alpha`` over the next ``warmup_epochs`` epochs, then stays."""
    if warmup_epochs <= 0:
        return float(target_alpha)
    if epoch < warmup_epochs:
        return 0.0
    ramp = min(1.0, (epoch - warmup_epochs + 1) / float(warmup_epochs))
    return float(target_alpha) * ramp


def main():
    args = parse_args()

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable; using CPU.")

    data = lasa_dataset.LASA(filename=args.shape, device=device)
    dim = data.dim
    fm_sampler, pair_sampler = build_lasa_samplers(data, device, window_length=2)

    hidden_dims = tuple(int(h) for h in args.hidden.split("-"))
    spatial_ode, sde = build_models(dim, hidden_dims, args.dt, device)

    composite = model.DecoupledFMImitationFlow(
        spatial_ode=spatial_ode, dynamics=sde, dim=dim, device=device
    ).to(device)

    optimizer = optim.Adam(
        list(spatial_ode.parameters()) + list(sde.parameters()), lr=args.lr
    )

    plot_dir = os.path.join(
        os.path.dirname(__file__), "lasa_fm_e2e_plots", args.shape, args.out_name
    )
    weights_dir = os.path.join(
        os.path.dirname(__file__), "lasa_fm_e2e_weights", args.shape
    )
    makedirs(plot_dir)
    makedirs(weights_dir)

    for epoch in range(args.epochs):
        alpha = alpha_schedule(epoch, args.alpha, args.alpha_warmup)

        spatial_ode.train()
        sde.train()

        running = {"loss": 0.0, "loss_fm": 0.0, "loss_sde": 0.0}
        for _ in range(args.steps_per_epoch):
            optimizer.zero_grad()

            x1 = fm_sampler.sample(args.fm_batch)
            l_fm = loss_fm(spatial_ode, x1)

            if alpha > 0.0:
                windows = pair_sampler.sample(args.pair_batch)
                z_pairs = encode_window(spatial_ode, windows, training=True)
                l_sde = loss_phase_c(sde, z_pairs, dt=args.dt)
                loss = l_fm + alpha * l_sde
                running["loss_sde"] += float(l_sde.detach())
            else:
                loss = l_fm

            loss.backward()
            optimizer.step()

            running["loss"] += float(loss.detach())
            running["loss_fm"] += float(l_fm.detach())

        n = max(args.steps_per_epoch, 1)
        avg_loss = running["loss"] / n
        avg_fm = running["loss_fm"] / n
        avg_sde = running["loss_sde"] / n if alpha > 0.0 else float("nan")

        with torch.no_grad():
            eigvals = sde.eigvals()
            max_re = float(eigvals.real.max())
            stable = bool((eigvals.real < 0).all())

        print(
            "[epoch {:04d}] alpha={:.3g} loss={:.4f} fm={:.4f} sde={:.4f} "
            "stable={} max_re={:.4f}".format(
                epoch, alpha, avg_loss, avg_fm, avg_sde, stable, max_re
            )
        )

        do_vis = (epoch % args.vis_every == 0) or (epoch == args.epochs - 1)
        do_ckpt = (epoch % args.ckpt_every == 0) or (epoch == args.epochs - 1)

        if do_vis:
            with torch.no_grad():
                composite.eval()
                visualize_2d_generated_trj(
                    data.train_data,
                    composite,
                    device,
                    fig_number=2,
                    save_path=os.path.join(
                        plot_dir, "epoch_{:04d}_traj2d.png".format(epoch)
                    ),
                )
                visualize_latent_distribution(
                    data.train_data,
                    composite,
                    device,
                    fig_number=1,
                    save_path=os.path.join(
                        plot_dir, "epoch_{:04d}_latent.png".format(epoch)
                    ),
                )
                visualize_vector_field(
                    data.train_data,
                    composite,
                    device,
                    fig_number=3,
                    save_path=os.path.join(
                        plot_dir, "epoch_{:04d}_vector_field.png".format(epoch)
                    ),
                )

        if do_ckpt:
            ckpt_path = os.path.join(
                weights_dir, "{}_epoch{:04d}.pt".format(args.out_name, epoch)
            )
            torch.save(
                {
                    "shape": args.shape,
                    "epoch": epoch,
                    "dim": dim,
                    "hidden_dims": list(hidden_dims),
                    "dt": args.dt,
                    "alpha": alpha,
                    "alpha_target": args.alpha,
                    "alpha_warmup": args.alpha_warmup,
                    "spatial_ode_state_dict": spatial_ode.state_dict(),
                    "sde_state_dict": sde.state_dict(),
                    "lasa_mean": data.mean,
                    "lasa_std": data.std,
                },
                ckpt_path,
            )
            print("Saved checkpoint: {}".format(ckpt_path))


if __name__ == "__main__":
    main()
