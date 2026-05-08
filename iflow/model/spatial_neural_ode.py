"""
Spatial Neural ODE for the decoupled ImitationFlow pipeline (Phase A / B).

This module defines the `tau`-parameterized Neural ODE that maps a base Gaussian
distribution to the spatial distribution of expert demonstrations. It is trained
with simulation-free Conditional Flow Matching (no log-determinant integration),
and used at inference / Phase B to encode (`tau: 1 -> 0`) and decode
(`tau: 0 -> 1`) workspace points.

A `DummyLinearPredictor` is co-located here to support the Phase A temporal
regularization loss `L_Reg = MSE(z_{t+1} - z_t, A_hat z_t * dt)`.
"""
import torch
import torch.nn as nn

from torchdiffeq import odeint, odeint_adjoint

from iflow.model.cflows.odefunc import ODEnet


class SpatialVelocityNet(nn.Module):
    r"""
    Velocity field :math:`v_\theta(x, \tau)` for the spatial Neural ODE.

    Wraps `iflow.model.cflows.odefunc.ODEnet` so the same MLP family used by the
    repository's CNF code can be reused. Time `tau` is concatenated to the
    spatial input internally by the chosen `layer_type` (`"concat"` by default).
    """

    def __init__(self, dim, hidden_dims=(64, 64, 64), layer_type="concat", nonlinearity="tanh"):
        super().__init__()
        self.dim = dim
        self.net = ODEnet(
            hidden_dims=tuple(hidden_dims),
            input_shape=(dim,),
            strides=None,
            conv=False,
            layer_type=layer_type,
            nonlinearity=nonlinearity,
        )

    def forward(self, tau, x):
        if not torch.is_tensor(tau):
            tau = torch.tensor(tau, dtype=x.dtype, device=x.device)
        else:
            tau = tau.to(dtype=x.dtype, device=x.device)
        return self.net(tau, x)


class NeuralODEFlow(nn.Module):
    r"""
    Thin Neural ODE wrapper around `torchdiffeq.odeint`.

    Forward (`decode`) integrates the velocity field from :math:`\tau=0` to
    :math:`\tau=1`; backward (`encode`) integrates from :math:`\tau=1` to
    :math:`\tau=0`. Unlike `iflow.model.cflows.cnf.CNF`, no log-density state is
    carried, so training the velocity net through this module is purely a
    regression problem when used under Conditional Flow Matching.
    """

    def __init__(
        self,
        velocity_net,
        solver="dopri5",
        atol=1e-5,
        rtol=1e-5,
        train_solver="euler",
        train_solver_steps=20,
        use_adjoint=False,
    ):
        super().__init__()
        self.velocity_net = velocity_net
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.train_solver = train_solver
        self.train_solver_steps = max(int(train_solver_steps), 1)
        self.use_adjoint = use_adjoint

    @property
    def dim(self):
        return self.velocity_net.dim

    def _integrate(self, x0, integration_times, training):
        if training:
            method = self.train_solver
            options = None
            if method in ("euler", "rk4", "midpoint"):
                step_size = float(
                    abs(integration_times[-1] - integration_times[0]) / self.train_solver_steps
                )
                options = {"step_size": step_size}
            kwargs = dict(method=method)
            if options is not None:
                kwargs["options"] = options
        else:
            kwargs = dict(method=self.solver, atol=self.atol, rtol=self.rtol)

        integrator = odeint_adjoint if (self.use_adjoint and training) else odeint
        states = integrator(self.velocity_net, x0, integration_times, **kwargs)
        return states

    def decode(self, x0, tau_grid=None, training=None):
        training = self.training if training is None else training
        if tau_grid is None:
            tau_grid = torch.tensor([0.0, 1.0], dtype=x0.dtype, device=x0.device)
        else:
            tau_grid = tau_grid.to(dtype=x0.dtype, device=x0.device)
        states = self._integrate(x0, tau_grid, training)
        return states[-1]

    def encode(self, x1, tau_grid=None, training=None):
        training = self.training if training is None else training
        if tau_grid is None:
            tau_grid = torch.tensor([1.0, 0.0], dtype=x1.dtype, device=x1.device)
        else:
            tau_grid = tau_grid.to(dtype=x1.dtype, device=x1.device)
        states = self._integrate(x1, tau_grid, training)
        return states[-1]

    def encode_batched(self, points, chunk_size=1024, training=False):
        was_training = self.training
        self.train(training)
        try:
            outs = []
            for start in range(0, points.shape[0], chunk_size):
                chunk = points[start : start + chunk_size]
                outs.append(self.encode(chunk, training=training))
            return torch.cat(outs, dim=0)
        finally:
            self.train(was_training)

    def forward(self, x, reverse=False):
        return self.encode(x) if reverse else self.decode(x)


class DummyLinearPredictor(nn.Module):
    r"""
    Unconstrained linear predictor :math:`\hat{A} \in \mathbb{R}^{d \times d}`
    used for the Phase A temporal regularization term

    .. math::
        \mathcal{L}_{\text{Reg}} = \mathrm{MSE}(z_{t+1} - z_t,\, \hat{A} z_t\, \Delta t).

    This is intentionally *not* the Hurwitz parameterization from Phase C; the
    point is only to penalize latent trajectories that cannot be approximated by
    *any* linear system. The diagonal is initialized slightly negative so the
    linear baseline starts as a mildly contractive map.
    """

    def __init__(self, dim, init_diag=-1.0):
        super().__init__()
        self.dim = dim
        A0 = torch.eye(dim) * float(init_diag) * 0.1
        self.A_hat = nn.Parameter(A0)

    def forward(self, z):
        return z @ self.A_hat.T

    def predict_increment(self, z, dt):
        return self(z) * float(dt)
