"""
Load a train_lasa checkpoint and save trajectory / latent / vector-field figures (no training).

Dynamics hyperparameters (dt, T_to_stable, Tanh vs Linear) must match how the checkpoint was trained;
train_lasa uses TanhStochasticDynamics with dt=0.01 and T_to_stable=2.5.
"""
import argparse
import os

import torch

from iflow import model
from iflow.dataset import lasa_dataset
from iflow.utils import makedirs
from iflow.visualization import (
    visualize_2d_generated_trj,
    visualize_latent_distribution,
    visualize_vector_field,
)


def main_layer(dim):
    return model.CouplingLayer(dim)


def create_flow_seq(dim, depth):
    chain = []
    for _ in range(depth):
        chain.append(main_layer(dim))
        chain.append(model.RandomPermutation(dim))
        chain.append(model.LULinear(dim))
    chain.append(main_layer(dim))
    return model.SequentialFlow(chain)


def parse_args():
    p = argparse.ArgumentParser(description="LASA generation + visualization from checkpoint")
    p.add_argument("checkpoint", type=str, help="Path to .pt from train_lasa")
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for PNGs (default: examples/lasa_generated/<ckpt_basename>)",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=("cpu", "cuda"),
        help="Device for inference",
    )
    return p.parse_args()


def main():
    args = parse_args()
    ckpt_path = os.path.abspath(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    shape = ckpt["shape"]
    dim = int(ckpt["dim"])
    depth = int(ckpt["depth"])
    epoch = int(ckpt.get("epoch", 0))
    prefix = "epoch_{:04d}_".format(epoch)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable; using CPU.")

    if args.out_dir is not None:
        out_dir = os.path.abspath(args.out_dir)
    else:
        base = os.path.splitext(os.path.basename(ckpt_path))[0]
        out_dir = os.path.join(os.path.dirname(__file__), "lasa_generated", base)
    makedirs(out_dir)

    dynamics = model.TanhStochasticDynamics(dim, dt=0.01, T_to_stable=2.5)
    flow = create_flow_seq(dim, depth)
    iflow = model.ContinuousDynamicFlow(dynamics=dynamics, model=flow, dim=dim).to(device)
    iflow.load_state_dict(ckpt["iflow_state_dict"], strict=True)

    data = lasa_dataset.LASA(filename=shape, device=device)

    with torch.no_grad():
        iflow.eval()
        visualize_2d_generated_trj(
            data.train_data,
            iflow,
            device,
            fig_number=10,
            save_path=os.path.join(out_dir, prefix + "traj2d.png"),
        )
        visualize_latent_distribution(
            data.train_data,
            iflow,
            device,
            fig_number=11,
            save_path=os.path.join(out_dir, prefix + "latent.png"),
        )
        visualize_vector_field(
            data.train_data,
            iflow,
            device,
            fig_number=12,
            save_path=os.path.join(out_dir, prefix + "vector_field.png"),
        )

    print("Wrote figures to {}".format(out_dir))


if __name__ == "__main__":
    main()
