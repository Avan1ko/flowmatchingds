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
import time

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
    p.add_argument("--shape", type=str, default="Angle")
    p.add_argument("--device", type=str, default="cpu", choices=("cpu", "cuda"))
    p.add_argument("--macro-epochs", type=int, default=35)
    p.add_argument("--phase-a-iters", type=int, default=400)
    p.add_argument("--phase-c-iters", type=int, default=200)
    p.add_argument("--fm-batch", type=int, default=512)
    p.add_argument("--reg-batch", type=int, default=16)
    p.add_argument("--reg-window", type=int, default=8)
    p.add_argument("--reg-every-k", type=int, default=10)
    p.add_argument("--lambda-reg", type=float, default=0.1)
    p.add_argument("--lambda-warmup", type=int, default=2,
                   help="Macro-epochs with lambda_reg=0 before ramping in")
    p.add_argument("--dt", type=float, default=0.03)
    p.add_argument("--lr-spatial", type=float, default=1e-3)
    p.add_argument("--lr-sde", type=float, default=1e-2)
    p.add_argument("--hidden", type=str, default="64-64-64")
    p.add_argument("--encode-chunk", type=int, default=2048)
    p.add_argument("--out-name", type=str, default="lasa_fm_hybrid")
    p.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save checkpoints and plots every N macro-epochs (and always on the last)",
    )
    return p.parse_args()


def build_models(dim, hidden_dims, dt, device):
    velocity_net = model.SpatialVelocityNet(dim=dim, hidden_dims=hidden_dims, layer_type="concat", nonlinearity="elu")
    spatial_ode = model.NeuralODEFlow(velocity_net=velocity_net, train_solver="euler", train_solver_steps=50)
    dummy = model.DummyLinearPredictor(dim=dim, init_diag=-1.0)
    sde = model.StableLinearSDE(dim=dim, dt=dt)

    sde.predict_increment = lambda z, dt_val: (
        sde(z)[0] if isinstance(sde(z), (tuple, list)) else sde(z)
    ) * dt_val

    spatial_ode.to(device)
    dummy.to(device)
    sde.to(device)
    return spatial_ode, dummy, sde

def compute_final_metrics(expert_trajectories, model, device, threshold=0.5):
    """
    Calculates reproduction and stability metrics at the end of training.
    """
    mse_list, dtw_list, jerk_list = [], [], []
    success_count = 0
    
    def simple_dtw(s1, s2):
        """Basic O(N^2) DTW implementation for trajectory comparison."""
        n, m = len(s1), len(s2)
        cost_mat = np.zeros((n + 1, m + 1))
        cost_mat[1:, 0], cost_mat[0, 1:] = np.inf, np.inf
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                dist = np.linalg.norm(s1[i-1] - s2[j-1])
                cost_mat[i, j] = dist + min(cost_mat[i-1, j], cost_mat[i, j-1], cost_mat[i-1, j-1])
        return cost_mat[n, m]

    print("\n--- Final Performance Metrics ---")
    for i, expert in enumerate(expert_trajectories):
        expert_np = np.asarray(expert)
        y0 = torch.from_numpy(expert_np[0:1]).float().to(device)
        goal_true = expert_np[-1]
        
        with torch.no_grad():
            # Generate a trajectory of the same length as the expert
            gen_trj = model.generate_trj(y0, T=len(expert_np)).cpu().numpy()
        
        # 1. Root Mean Squared Error
        mse = np.sqrt(np.mean(np.linalg.norm(expert_np - gen_trj, axis=1)**2))
        mse_list.append(mse)
        
        # 2. Dynamic Time Warping (DTW)
        dtw_val = simple_dtw(expert_np, gen_trj)
        dtw_list.append(dtw_val)
        
        # 3. Success Rate (Distance to Goal)
        final_dist = np.linalg.norm(gen_trj[-1] - goal_true)
        if final_dist < threshold:
            success_count += 1
            
        # 4. Smoothness: Mean Jerk (Third derivative)
        # Calculated via finite differences: jerk = d^3x / dt^3
        vel = np.diff(gen_trj, axis=0)
        acc = np.diff(vel, axis=0)
        jerk = np.diff(acc, axis=0)
        mean_jerk = np.mean(np.linalg.norm(jerk, axis=1))
        jerk_list.append(mean_jerk)

    results = {
        "Avg RMSE": np.mean(mse_list),
        "Avg DTW": np.mean(dtw_list),
        "Success Rate": (success_count / len(expert_trajectories)) * 100,
        "Mean Jerk": np.mean(jerk_list)
    }
    
    for k, v in results.items():
        print(f"{k:15}: {v:.4f}")
    return results

def main():
    args = parse_args()

    start_train_time = time.time()
    epoch_times = []

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

    goals_x = []
    for tr in data.train_data:
        goals_x.append(tr[-1:])
    goals_x = torch.from_numpy(np.concatenate(goals_x, axis=0)).float().to(device)

    global_step = 0
    for macro_epoch in range(args.macro_epochs):
        epoch_start = time.time()
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
            #Using Vector Field (No ODE Solver needed)
            t_anchor = torch.rand(goals_x.shape[0], 1).to(device)
            x_t_anchor = t_anchor * goals_x
            u_t_anchor = goals_x
            v_pred_anchor = spatial_ode.velocity_net(t_anchor, x_t_anchor)
            loss_goal = torch.nn.functional.mse_loss(v_pred_anchor, u_t_anchor)
            
            #Using MSE (Solving ODE)
            # z_goals = spatial_ode.encode(goals_x)
            # loss_goal = torch.nn.functional.mse_loss(z_goals, torch.zeros_like(z_goals))
            lambda_goal = 1.0
            total_goal_loss = lambda_goal * loss_goal
            total_goal_loss.backward()
            optim_a.step()
            optim_a.zero_grad()

            running["loss_goal"] = running.get("loss_goal", 0.0) + loss_goal.item()
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
    total_train_time = time.time() - start_train_time
    # avg_epoch_time = sum(epoch_times) / len(epoch_times)
    print("\nTraining complete. Running final evaluation...")
    print(f"Total Training Time: {total_train_time:.2f} seconds")
    # print(f"Average Macro-Epoch Time: {avg_epoch_time:.2f} seconds")
    composite.eval()
    final_metrics = compute_final_metrics(data.train_data, composite, device)



if __name__ == "__main__":
    main()