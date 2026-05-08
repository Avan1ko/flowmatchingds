"""
Plotting helpers for side-by-side model comparisons.

Each function operates on a *list* of model results (typically two: FM-hybrid
and CNF baseline) so the same call site can render a single figure with both
methods. Models are identified by their string label.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt


_DEFAULT_COLORS = ("tab:orange", "tab:blue", "tab:green", "tab:red")


def _rollout_for_plot(model, val_trajs, device):
    out = []
    for trj in val_trajs:
        n = trj.shape[0]
        y0 = torch.from_numpy(trj[0, :][None, :]).float().to(device)
        out.append(model.generate_trj(y0, T=n).detach().cpu().numpy())
    return out


def plot_trajectories_side_by_side(val_trajs, models, labels, device, save_path=None, fig_number=20):
    r"""
    Overlay expert trajectories with each model's rollouts on a single 2D axis.

    ``models`` and ``labels`` must have matching length.
    """
    plt.figure(fig_number).clf()
    fig = plt.figure(figsize=(8, 8), num=fig_number)
    ax = plt.gca()

    for trj in val_trajs:
        ax.plot(trj[:, 0], trj[:, 1], color="black", alpha=0.6, linewidth=1.2)
    expert_handle, = ax.plot([], [], color="black", label="expert")

    handles = [expert_handle]
    for color, model, label in zip(_DEFAULT_COLORS, models, labels):
        preds = _rollout_for_plot(model, val_trajs, device)
        for p in preds:
            ax.plot(p[:, 0], p[:, 1], color=color, alpha=0.7, linewidth=1.0)
        h, = ax.plot([], [], color=color, label=label)
        handles.append(h)

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("Expert vs generated trajectories")
    ax.legend(handles=handles, loc="best")
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
    else:
        plt.draw()
        plt.pause(0.001)


def plot_metric_bars(results, labels, metric_keys, save_path=None, fig_number=21):
    r"""
    Grouped bar chart of scalar metrics for each model.

    ``results`` is a list of dicts (one per model) keyed by metric name.
    """
    n_metrics = len(metric_keys)
    n_models = len(results)
    width = 0.8 / max(n_models, 1)
    x = np.arange(n_metrics)

    plt.figure(fig_number).clf()
    fig, ax = plt.subplots(figsize=(max(8, 1.4 * n_metrics), 5), num=fig_number)
    for i, (res, lab) in enumerate(zip(results, labels)):
        vals = [float(res.get(k, np.nan)) for k in metric_keys]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width=width, label=lab,
               color=_DEFAULT_COLORS[i % len(_DEFAULT_COLORS)])
        for xi, v in zip(x + i * width - 0.4 + width / 2, vals):
            if np.isfinite(v):
                ax.text(xi, v, "{:.3g}".format(v), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_keys, rotation=20, ha="right")
    ax.set_ylabel("metric value (lower is better)")
    ax.set_title("Quantitative comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
    else:
        plt.draw()
        plt.pause(0.001)


def plot_final_point_errors(results, labels, save_path=None, fig_number=22):
    r"""
    Per-demonstration final-point error scatter (one column per model).
    """
    plt.figure(fig_number).clf()
    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(results)), 5), num=fig_number)
    for i, (res, lab) in enumerate(zip(results, labels)):
        per_demo = res["_final_point_detail"]["per_demo"]
        x = np.full(len(per_demo), i, dtype=float) + np.random.uniform(-0.1, 0.1, len(per_demo))
        ax.scatter(x, per_demo, color=_DEFAULT_COLORS[i % len(_DEFAULT_COLORS)],
                   alpha=0.7, s=30, label=lab)
        ax.hlines(np.mean(per_demo), i - 0.25, i + 0.25, color="black", linewidth=2)
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("||y_pred[-1] - y_expert[-1]||")
    ax.set_title("Final-point error per demo (black bars = mean)")
    ax.grid(axis="y", alpha=0.3)
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
    else:
        plt.draw()
        plt.pause(0.001)


def plot_off_distribution_heatmap(val_trajs, results, labels, save_path=None, fig_number=23):
    r"""
    One panel per model showing the off-distribution success grid in the
    ``y0`` plane. Successes are green, failures are red, and the goal /
    expert paths are overlaid in black.
    """
    n_models = len(results)
    plt.figure(fig_number).clf()
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6),
                             num=fig_number, squeeze=False)
    for i, (res, lab) in enumerate(zip(results, labels)):
        ax = axes[0, i]
        ood = res["_ood_detail"]
        xs = ood["xs"]
        ys = ood["ys"]
        success = ood["success_mask"]
        XX, YY = np.meshgrid(xs, ys, indexing="ij")
        ax.scatter(
            XX[success], YY[success], color="tab:green", s=60,
            marker="o", alpha=0.9, label="success",
        )
        ax.scatter(
            XX[~success], YY[~success], color="tab:red", s=60,
            marker="x", alpha=0.9, label="fail",
        )
        for trj in val_trajs:
            ax.plot(trj[:, 0], trj[:, 1], color="black", alpha=0.4, linewidth=1.0)
        goal = ood["goal"]
        ax.scatter(goal[0], goal[1], color="black", s=160, marker="*", label="goal")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title("{} (success rate = {:.2f})".format(lab, ood["rate"]))
        ax.legend(loc="best", fontsize=8)
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
    else:
        plt.draw()
        plt.pause(0.001)


def plot_latent_residual_histogram(results, labels, save_path=None, fig_number=24, bins=40):
    r"""
    Histogram of per-pair latent linear-fit residuals (log scale on x).
    """
    plt.figure(fig_number).clf()
    fig, ax = plt.subplots(figsize=(8, 5), num=fig_number)

    all_vals = np.concatenate([res["_latent_detail"]["per_pair"] for res in results])
    eps = max(float(all_vals.min()), 1e-12) * 0.5
    all_vals = np.maximum(all_vals, eps)
    edges = np.logspace(np.log10(all_vals.min()), np.log10(all_vals.max() + eps), bins)

    for i, (res, lab) in enumerate(zip(results, labels)):
        per_pair = np.maximum(res["_latent_detail"]["per_pair"], eps)
        ax.hist(per_pair, bins=edges, alpha=0.5,
                color=_DEFAULT_COLORS[i % len(_DEFAULT_COLORS)],
                label="{} (mean={:.3g})".format(lab, res["latent_linear_residual"]))
    ax.set_xscale("log")
    ax.set_xlabel("per-pair residual ||dz - A_ls z||^2")
    ax.set_ylabel("count")
    ax.set_title("Latent linear-fit residual distribution (lower / leftward is better)")
    ax.legend()
    ax.grid(alpha=0.3)
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
    else:
        plt.draw()
        plt.pause(0.001)


def plot_wasserstein_projections(results, labels, save_path=None, fig_number=25, bins=20):
    r"""
    Histogram of per-projection 1D Wasserstein distances; tighter / leftward
    distributions indicate a better spatial match.
    """
    plt.figure(fig_number).clf()
    fig, ax = plt.subplots(figsize=(8, 5), num=fig_number)
    for i, (res, lab) in enumerate(zip(results, labels)):
        vals = np.asarray(res["_wasserstein_detail"]["per_projection"])
        ax.hist(vals, bins=bins, alpha=0.5,
                color=_DEFAULT_COLORS[i % len(_DEFAULT_COLORS)],
                label="{} (mean={:.3g})".format(lab, vals.mean()))
    ax.set_xlabel("1D Wasserstein distance per random projection")
    ax.set_ylabel("count")
    ax.set_title("Sliced Wasserstein components (lower is better)")
    ax.legend()
    ax.grid(alpha=0.3)
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
    else:
        plt.draw()
        plt.pause(0.001)
