"""
Side-by-side evaluation of the decoupled FM-hybrid model vs the original
CNF / discrete-NF baseline on a single LASA shape.

Loads one checkpoint per model, runs the full IROS metric suite plus the new
``comparison_metrics`` (final-point error, off-distribution success rate,
latent linear-fit residual, sliced Wasserstein), and writes:

* ``metrics.csv`` -- one row per (shape, model, metric)
* ``trajectories.png``, ``metrics_bar.png``, ``final_point.png``,
  ``off_distribution.png``, ``latent_residual.png``, ``wasserstein.png``

into the chosen output directory.
"""
import argparse
import csv
import os

import torch

from iflow import model
from iflow.dataset import lasa_dataset
from iflow.test_measures import evaluate_model
from iflow.utils import makedirs
from iflow.visualization import (
    plot_final_point_errors,
    plot_latent_residual_histogram,
    plot_metric_bars,
    plot_off_distribution_heatmap,
    plot_trajectories_side_by_side,
    plot_wasserstein_projections,
)


SCALAR_METRIC_KEYS = (
    "mean_l2",
    "frechet",
    "dtw",
    "swept_area",
    "area_between",
    "final_point_error",
    "sliced_wasserstein",
    "latent_linear_residual",
)


def parse_args():
    p = argparse.ArgumentParser(description="FM-hybrid vs CNF evaluation on LASA")
    p.add_argument("--shape", type=str, required=True,
                   help="LASA shape name (must match both checkpoints)")
    p.add_argument("--cnf-ckpt", type=str, required=True,
                   help="Path to a checkpoint produced by examples/train_lasa.py")
    p.add_argument("--fm-ckpt", type=str, required=True,
                   help="Path to a checkpoint produced by examples/train_lasa_fm_hybrid.py")
    p.add_argument("--device", type=str, default="cpu", choices=("cpu", "cuda"))
    p.add_argument("--out-dir", type=str, default=None,
                   help="Output directory (defaults to examples/lasa_eval/<shape>)")
    p.add_argument("--ood-grid", type=int, default=15,
                   help="Side length of the off-distribution start-point grid")
    p.add_argument("--halo-scale", type=float, default=1.5,
                   help="Bounding-box scale for off-distribution starts")
    return p.parse_args()


def _build_cnf_main_layer(dim):
    return model.CouplingLayer(dim)


def _build_cnf_flow(dim, depth):
    chain = []
    for _ in range(depth):
        chain.append(_build_cnf_main_layer(dim))
        chain.append(model.RandomPermutation(dim))
        chain.append(model.LULinear(dim))
    chain.append(_build_cnf_main_layer(dim))
    return model.SequentialFlow(chain)


def load_cnf(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    dim = int(ckpt["dim"])
    depth = int(ckpt["depth"])
    dynamics = model.TanhStochasticDynamics(dim, dt=0.01, T_to_stable=2.5)
    flow = _build_cnf_flow(dim, depth)
    iflow = model.ContinuousDynamicFlow(dynamics=dynamics, model=flow, dim=dim).to(device)
    iflow.load_state_dict(ckpt["iflow_state_dict"], strict=True)
    iflow.eval()
    return iflow, ckpt


def load_fm(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    dim = int(ckpt["dim"])
    hidden_dims = tuple(int(h) for h in ckpt["hidden_dims"])
    dt = float(ckpt["dt"])

    velocity_net = model.SpatialVelocityNet(
        dim=dim, hidden_dims=hidden_dims, layer_type="concat", nonlinearity="tanh"
    )
    spatial_ode = model.NeuralODEFlow(
        velocity_net=velocity_net, train_solver="euler", train_solver_steps=20
    )
    sde = model.StableLinearSDE(dim=dim, dt=dt)

    spatial_ode.load_state_dict(ckpt["spatial_ode_state_dict"], strict=True)
    sde.load_state_dict(ckpt["sde_state_dict"], strict=True)
    spatial_ode.to(device)
    sde.to(device)

    composite = model.DecoupledFMImitationFlow(
        spatial_ode=spatial_ode, dynamics=sde, dim=dim, device=device,
    ).to(device)
    composite.eval()
    return composite, ckpt


def write_csv(path, shape, results, labels):
    rows = [["shape", "model", "metric", "value"]]
    for res, lab in zip(results, labels):
        for k in SCALAR_METRIC_KEYS:
            if k in res:
                rows.append([shape, lab, k, "{:.6f}".format(res[k])])
        rows.append([shape, lab, "off_distribution_success",
                     "{:.6f}".format(res["off_distribution_success"])])
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerows(rows)


def main():
    args = parse_args()
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable; using CPU.")

    if args.out_dir is None:
        args.out_dir = os.path.join(os.path.dirname(__file__), "lasa_eval", args.shape)
    makedirs(args.out_dir)

    data = lasa_dataset.LASA(filename=args.shape, device=device)

    print("Loading CNF baseline from {}".format(args.cnf_ckpt))
    cnf_model, _ = load_cnf(args.cnf_ckpt, device)
    print("Loading FM hybrid from   {}".format(args.fm_ckpt))
    fm_model, _ = load_fm(args.fm_ckpt, device)

    models = [fm_model, cnf_model]
    labels = ["FM_hybrid", "CNF"]

    results = []
    with torch.no_grad():
        for m, lab in zip(models, labels):
            print("Evaluating {} ...".format(lab))
            res = evaluate_model(
                data.train_data, m, device,
                n_grid=args.ood_grid, halo_scale=args.halo_scale,
            )
            print("  mean_l2={:.4f} frechet={:.4f} dtw={:.4f} fpe={:.4f}".format(
                res["mean_l2"], res["frechet"], res["dtw"], res["final_point_error"]
            ))
            print("  ood_success={:.3f} latent_lin_res={:.4g} sliced_W={:.4f}".format(
                res["off_distribution_success"],
                res["latent_linear_residual"],
                res["sliced_wasserstein"],
            ))
            results.append(res)

    csv_path = os.path.join(args.out_dir, "metrics.csv")
    write_csv(csv_path, args.shape, results, labels)
    print("Wrote {}".format(csv_path))

    with torch.no_grad():
        plot_trajectories_side_by_side(
            data.train_data, models, labels, device,
            save_path=os.path.join(args.out_dir, "trajectories.png"),
        )
        plot_metric_bars(
            results, labels, list(SCALAR_METRIC_KEYS),
            save_path=os.path.join(args.out_dir, "metrics_bar.png"),
        )
        plot_final_point_errors(
            results, labels,
            save_path=os.path.join(args.out_dir, "final_point.png"),
        )
        plot_off_distribution_heatmap(
            data.train_data, results, labels,
            save_path=os.path.join(args.out_dir, "off_distribution.png"),
        )
        plot_latent_residual_histogram(
            results, labels,
            save_path=os.path.join(args.out_dir, "latent_residual.png"),
        )
        plot_wasserstein_projections(
            results, labels,
            save_path=os.path.join(args.out_dir, "wasserstein.png"),
        )

    print("Wrote figures to {}".format(args.out_dir))


if __name__ == "__main__":
    main()
