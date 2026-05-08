"""
Sampling helpers for the decoupled Flow Matching trainer.

Phase A needs two different views of the same expert demonstrations:

* **FM mode** -- pooled, i.i.d. spatial samples ``x1`` drawn from the union of
  all expert points across all timesteps and demos.
* **Reg mode** -- short *ordered* windows ``y_{t:t+L}`` from a single
  trajectory (preserving the physical time ordering needed to compute
  :math:`\Delta z_t`).

Both views are read directly from the normalized trajectory list produced by
``iflow.dataset.lasa_dataset.LASA``.
"""
import numpy as np
import torch


class FMPointSampler:
    r"""
    Pools all expert points and serves uniform i.i.d. mini-batches for the
    Conditional Flow Matching loss.
    """

    def __init__(self, trajs, device):
        flat = np.concatenate([np.asarray(tr) for tr in trajs], axis=0)
        self.points = torch.from_numpy(flat).float().to(device)
        self.device = device
        self.dim = self.points.shape[1]

    def __len__(self):
        return self.points.shape[0]

    def sample(self, batch_size):
        idx = torch.randint(0, self.points.shape[0], (batch_size,), device=self.device)
        return self.points[idx]


class TrajectoryWindowSampler:
    r"""
    Samples short ordered windows ``y_{t:t+L}`` for the temporal regularization
    loss. Each call returns a tensor of shape ``(B_reg, L, dim)`` with the
    physical-time ordering preserved along the ``L`` axis.
    """

    def __init__(self, trajs, device, window_length=8):
        self.trajs = [torch.from_numpy(np.asarray(tr)).float().to(device) for tr in trajs]
        self.device = device
        self.window_length = int(window_length)
        self.dim = self.trajs[0].shape[1]
        self._max_starts = [max(tr.shape[0] - self.window_length, 0) for tr in self.trajs]
        if all(m == 0 for m in self._max_starts):
            raise ValueError(
                "window_length={} exceeds every trajectory length".format(self.window_length)
            )

    def sample(self, batch_size):
        windows = []
        for _ in range(batch_size):
            i = np.random.randint(len(self.trajs))
            while self._max_starts[i] == 0:
                i = np.random.randint(len(self.trajs))
            t = np.random.randint(0, self._max_starts[i] + 1)
            windows.append(self.trajs[i][t : t + self.window_length])
        return torch.stack(windows, dim=0)


def build_lasa_samplers(lasa_dataset, device, window_length=8):
    fm = FMPointSampler(lasa_dataset.train_data, device)
    reg = TrajectoryWindowSampler(lasa_dataset.train_data, device, window_length=window_length)
    return fm, reg
