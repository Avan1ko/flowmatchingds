"""
Globally asymptotically stable linear SDE for Phase C.

Reparameterizes :math:`A_\phi` as :math:`A = S - D` with :math:`S = M - M^\top`
skew-symmetric and :math:`D = L L^\top + \epsilon I` symmetric positive
definite, so the deterministic linear system :math:`\dot z = A z` is provably
Hurwitz: for any nonzero :math:`z`,

.. math::
    z^\top A z = z^\top S z - z^\top D z = -z^\top D z < 0.

The diffusion is parameterized as a learnable lower-triangular matrix
:math:`K_\phi`; the Brownian increment per step is :math:`K_\phi\,\Delta B_t`,
matching the existing `DynamicModel` Euler--Maruyama style of stepping.
"""
import math

import numpy as np
import torch
import torch.distributions as tdist
import torch.nn as nn

from iflow.model.dynamics.generic_dynamic import DynamicModel


def _tril_mask(dim, device=None):
    return torch.tril(torch.ones(dim, dim, device=device))


class StableLinearSDE(DynamicModel):
    r"""
    Linear stochastic dynamics with a guaranteed-Hurwitz drift.

    Drift: :math:`f(z) = A_\phi z` where :math:`A_\phi = (M - M^\top) - (L L^\top + \epsilon I)`.
    Diffusion: :math:`g(z) = K_\phi` (state-independent, lower-triangular).

    `velocity` returns :math:`A z`, so `step_forward` / `evolve` /
    `generate_trj` inherited from `DynamicModel` work without modification, with
    the diffusion term `var = K K^\top` matching the convention used by
    `DynamicModel.step_forward(noise=True)` (`var * dt`).
    """

    def __init__(
        self,
        dim,
        device=None,
        dt=0.01,
        requires_grad=True,
        epsilon=1e-3,
        init_decay=1.0,
        init_noise_std=0.1,
    ):
        super().__init__(dim, device, dt, requires_grad)

        self.epsilon = float(epsilon)

        M0 = torch.zeros(dim, dim)
        self.M = nn.Parameter(M0)

        L0 = torch.eye(dim) * math.sqrt(max(init_decay - epsilon, 1e-6))
        self.L_raw = nn.Parameter(L0)

        K0 = torch.eye(dim) * float(init_noise_std)
        self.K_raw = nn.Parameter(K0)

        self.register_buffer("_tril_mask_dim", _tril_mask(dim))

        if device is not None:
            self.to(device)

        if not requires_grad:
            for p in (self.M, self.L_raw, self.K_raw):
                p.requires_grad_(False)

    @property
    def L(self):
        return self.L_raw * self._tril_mask_dim

    @property
    def K(self):
        return self.K_raw * self._tril_mask_dim

    @property
    def S(self):
        return self.M - self.M.T

    @property
    def D(self):
        L = self.L
        return L @ L.T + self.epsilon * torch.eye(self.dim, device=L.device, dtype=L.dtype)

    @property
    def A(self):
        return self.S - self.D

    @property
    def var(self):
        K = self.K
        return K @ K.T

    def velocity(self, x):
        return x @ self.A.T

    def first_Taylor_dyn(self, x):
        return torch.cat(x.shape[0] * [self.A[None, ...]])

    def eigvals(self):
        return torch.linalg.eigvals(self.A.detach())

    def is_stable(self, tol=0.0):
        return bool((self.eigvals().real < tol).all())

    def stationary_covariance(self, n_iter=200, tol=1e-8):
        r"""
        Solve the continuous-time Lyapunov equation :math:`A P + P A^\top + Q = 0`
        with :math:`Q = K K^\top` by fixed-point iteration on the discrete-time
        analogue (sufficient for sanity checks, not high-precision use).
        """
        A = self.A.detach()
        Q = self.var.detach()
        I = torch.eye(self.dim, device=A.device, dtype=A.dtype)
        Ad = I + A * self.dt
        Qd = Q * self.dt
        P = torch.zeros_like(A)
        for _ in range(n_iter):
            P_new = Ad @ P @ Ad.T + Qd
            if torch.linalg.norm(P_new - P) < tol:
                P = P_new
                break
            P = P_new
        return P

    def compute_stable_log_px(self, x_n):
        P = self.stationary_covariance()
        P = P + 1e-6 * torch.eye(self.dim, device=P.device, dtype=P.dtype)
        mu = torch.zeros_like(x_n)
        dist = tdist.MultivariateNormal(loc=mu, covariance_matrix=P)
        return dist.log_prob(x_n)
