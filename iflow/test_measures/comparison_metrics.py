"""
Extra metrics for comparing two trajectory generators on the same dataset.

These complement `iros_evaluation` from `trajectory_metrics.py` with quantities
that highlight differences between the discrete-NF / CNF baseline and the
decoupled Flow-Matching + stable SDE pipeline:

* :func:`final_point_error`        -- distance of the rolled-out final point
                                       from the expert end-point.
* :func:`off_distribution_success` -- fraction of out-of-distribution starts
                                       that converge to the goal.
* :func:`latent_linear_fit_residual` -- residual of a least-squares linear fit
                                       to the encoded latent trajectories
                                       (proxy for "how Hurwitz-friendly" the
                                       latent space is).
* :func:`sample_wasserstein`       -- 1D sliced 2-Wasserstein between
                                       generated points and expert points
                                       (no external `pot` dependency required).

All functions accept any model that exposes the same `generate_trj(y0, T)` /
`forward(y) -> (z, _)` surface as the existing `ContinuousDynamicFlow`, which
includes the new :class:`iflow.model.DecoupledFMImitationFlow`.
"""
import numpy as np
import torch


def _rollout(model, val_trajs, device, T_factor=1.0):
    predicted = []
    for trj in val_trajs:
        n_trj = trj.shape[0]
        T = max(int(n_trj * T_factor), 2)
        y0 = torch.from_numpy(trj[0, :][None, :]).float().to(device)
        traj_pred = model.generate_trj(y0, T=T).detach().cpu().numpy()
        predicted.append(traj_pred)
    return predicted


def final_point_error(val_trajs, model, device):
    r"""
    Average L2 distance between the rolled-out final point and the expert
    final point (per-demonstration), with a list of per-demo errors for
    plotting.
    """
    errors = []
    predicted = _rollout(model, val_trajs, device)
    for ref, pred in zip(val_trajs, predicted):
        errors.append(float(np.linalg.norm(pred[-1] - ref[-1])))
    return {"mean": float(np.mean(errors)), "per_demo": errors}


def off_distribution_success(
    val_trajs,
    model,
    device,
    n_grid=15,
    halo_scale=1.5,
    horizon_factor=3.0,
    success_radius=0.2,
):
    r"""
    Sample starting points from a 2D bounding-box halo around the expert
    demonstrations, roll out for ``horizon_factor * mean_demo_length`` steps,
    and report the fraction whose final point is within ``success_radius`` of
    the expert end-point (in normalized coordinates -- LASA ``train_data`` is
    pre-normalized to zero-mean / unit-std).

    Returns the success rate plus the underlying grid + per-cell success mask
    so the caller can render a heatmap.
    """
    flat = np.concatenate([np.asarray(t) for t in val_trajs], axis=0)
    lo = flat.min(0)
    hi = flat.max(0)
    center = 0.5 * (lo + hi)
    span = (hi - lo) * halo_scale

    xs = np.linspace(center[0] - 0.5 * span[0], center[0] + 0.5 * span[0], n_grid)
    ys = np.linspace(center[1] - 0.5 * span[1], center[1] + 0.5 * span[1], n_grid)

    goal = np.mean([t[-1] for t in val_trajs], axis=0)

    mean_len = int(np.mean([t.shape[0] for t in val_trajs]))
    T = max(int(mean_len * horizon_factor), 2)

    success = np.zeros((n_grid, n_grid), dtype=bool)
    final_pts = np.zeros((n_grid, n_grid, 2), dtype=np.float32)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            y0 = torch.tensor([[float(x), float(y)]], dtype=torch.float32, device=device)
            traj = model.generate_trj(y0, T=T).detach().cpu().numpy()
            final_pts[i, j] = traj[-1]
            success[i, j] = bool(np.linalg.norm(traj[-1] - goal) < success_radius)

    return {
        "rate": float(success.mean()),
        "success_mask": success,
        "xs": xs,
        "ys": ys,
        "final_points": final_pts,
        "goal": goal,
    }


def _encode_traj(model, trj, device):
    """
    Encode a single (T, dim) trajectory into latent space using the model's
    forward pass. Both `ContinuousDynamicFlow` and `DecoupledFMImitationFlow`
    return ``(z, _)`` from ``forward``.
    """
    y = torch.from_numpy(np.asarray(trj)).float().to(device)
    z, _ = model(y)
    return z.detach().cpu().numpy()


def latent_linear_fit_residual(val_trajs, model, device):
    r"""
    Fit a single linear map :math:`A_{ls}` minimizing
    :math:`\sum_t \|\Delta z_t - A_{ls} z_t\|^2` across all encoded
    demonstrations and return the per-pair residual MSE plus the matrix.

    Lower is better and indicates that the spatial map produces latent
    trajectories that *can* be approximated by a single linear system -- which
    is exactly what Phase C needs for a faithful reconstruction.
    """
    Zs = []
    DZ = []
    for trj in val_trajs:
        z = _encode_traj(model, trj, device)
        Zs.append(z[:-1])
        DZ.append(z[1:] - z[:-1])
    Z = np.concatenate(Zs, axis=0)
    DZv = np.concatenate(DZ, axis=0)
    A_ls, residuals, rank, sv = np.linalg.lstsq(Z, DZv, rcond=None)
    pred = Z @ A_ls
    per_pair = ((pred - DZv) ** 2).sum(axis=1)
    return {
        "mse": float(per_pair.mean()),
        "per_pair": per_pair.astype(np.float32),
        "A_ls": A_ls.astype(np.float32),
    }


def sample_wasserstein(val_trajs, model, device, n_samples=2048, n_projections=64):
    r"""
    Sliced 2-Wasserstein between generated and expert *spatial* samples.

    Pools all expert points and rolls out one trajectory per demonstration,
    pools the generated samples, and averages the 1D Wasserstein distance of
    their projections along ``n_projections`` random unit directions.
    """
    expert = np.concatenate([np.asarray(t) for t in val_trajs], axis=0).astype(np.float32)
    generated = np.concatenate(
        [np.asarray(p) for p in _rollout(model, val_trajs, device, T_factor=1.0)],
        axis=0,
    ).astype(np.float32)

    # Drop diverged / NaN samples before computing the W distance: an
    # under-trained FM hybrid can produce inf / nan endpoints which would
    # otherwise overflow the projections.
    finite_mask = np.all(np.isfinite(generated), axis=1)
    if finite_mask.any():
        generated = generated[finite_mask]
    expert_max = float(np.max(np.abs(expert))) if expert.size else 1.0
    clip = max(expert_max * 50.0, 1.0)
    generated = np.clip(generated, -clip, clip)

    if expert.shape[0] > n_samples:
        idx = np.random.choice(expert.shape[0], n_samples, replace=False)
        expert = expert[idx]
    if generated.shape[0] > n_samples:
        idx = np.random.choice(generated.shape[0], n_samples, replace=False)
        generated = generated[idx]

    rng = np.random.default_rng(0)
    dim = expert.shape[1]
    dirs = rng.normal(size=(n_projections, dim))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12

    distances = []
    for d in dirs:
        e = np.sort(expert @ d)
        g = np.sort(generated @ d)
        n = min(e.shape[0], g.shape[0])
        e = e[:n]
        g = g[:n]
        distances.append(float(np.mean(np.abs(e - g))))
    return {"mean": float(np.mean(distances)), "per_projection": distances}


def collect_iros(val_trajs, model, device):
    """
    Return the standard IROS metrics (mean L2, Frechet, DTW, swept area, area
    between curves) as a dict, without printing.
    """
    from .trajectory_metrics import (
        squared_mean_error,
        mean_frechet_error,
        dtw_distance,
        mean_swept_error,
        area_between_error,
    )

    predicted = _rollout(model, val_trajs, device, T_factor=1.0)
    return {
        "mean_l2": squared_mean_error(val_trajs, predicted),
        "frechet": mean_frechet_error(val_trajs, predicted),
        "dtw": dtw_distance(val_trajs, predicted),
        "swept_area": mean_swept_error(val_trajs, predicted),
        "area_between": area_between_error(val_trajs, predicted),
    }


def evaluate_model(val_trajs, model, device, n_grid=15, halo_scale=1.5):
    r"""
    Run every comparison metric for one model and return a flat dict suitable
    for pickling / CSV export. Visualizations consume the nested-detail keys
    suffixed ``_detail``.
    """
    iros = collect_iros(val_trajs, model, device)
    fpe = final_point_error(val_trajs, model, device)
    ood = off_distribution_success(
        val_trajs, model, device, n_grid=n_grid, halo_scale=halo_scale
    )
    lin = latent_linear_fit_residual(val_trajs, model, device)
    swd = sample_wasserstein(val_trajs, model, device)

    return {
        **iros,
        "final_point_error": fpe["mean"],
        "off_distribution_success": ood["rate"],
        "latent_linear_residual": lin["mse"],
        "sliced_wasserstein": swd["mean"],
        "_final_point_detail": fpe,
        "_ood_detail": ood,
        "_latent_detail": lin,
        "_wasserstein_detail": swd,
    }
