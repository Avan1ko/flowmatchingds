"""
Phase D -- pushforward of the latent SDE to the observation space.

If :math:`y = h_\theta(z)` is the forward Neural ODE map, the workspace SDE is

.. math::
    dy = J_\theta(z) f_\phi(z)\,dt + J_\theta(z) g_\phi(z)\,dB_t

with :math:`J_\theta(z) = \partial h_\theta / \partial z`. The repository's
existing 2D CNF code already includes a row-by-row Jacobian helper
(`iflow.model.cflows.odefunc._get_minibatch_jacobian`); here we provide the two
recipes the plan describes:

1. **Explicit Jacobian** -- materialize :math:`J \in \mathbb{R}^{d\times d}`
   (cost: :math:`\mathcal{O}(d \cdot \text{NFE})` per state). Acceptable for
   ``d=2`` LASA.
2. **Matrix-free JVP** -- compute :math:`J v` directly via forward-mode AD on
   ``decode``. Cheaper for higher ``d`` because we only push the actual noise /
   drift directions, not every column.

Both helpers operate on a *single* state ``z`` (shape ``(dim,)`` or
``(B, dim)``). The repository typically operates batched, so both versions
support a leading batch dimension and use ``vmap`` where available.
"""
import torch


def _ensure_batched(z):
    if z.dim() == 1:
        return z.unsqueeze(0), True
    return z, False


def decode_jacobian(spatial_ode, z, create_graph=False):
    r"""
    Compute :math:`J = \partial \mathrm{decode}(z) / \partial z` row by row.

    For batched ``z`` returns a tensor of shape ``(B, dim, dim)``.
    """
    z_b, was_unbatched = _ensure_batched(z)
    z_b = z_b.detach().clone().requires_grad_(True)
    y = spatial_ode.decode(z_b, training=False)

    B, D = z_b.shape
    rows = []
    for j in range(D):
        grad_outputs = torch.zeros_like(y)
        grad_outputs[:, j] = 1.0
        (gj,) = torch.autograd.grad(
            outputs=y,
            inputs=z_b,
            grad_outputs=grad_outputs,
            retain_graph=True,
            create_graph=create_graph,
        )
        rows.append(gj.unsqueeze(1))
    J = torch.cat(rows, dim=1)
    return J.squeeze(0) if was_unbatched else J


def decode_jvp(spatial_ode, z, v, n_steps=None):
    r"""
    Forward-mode Jacobian-vector product :math:`J v` via ``torch.func.jvp``.

    Avoids materializing the full Jacobian -- one ODE solve per direction
    instead of one per output coordinate. Forward-mode AD does not compose well
    with adaptive solvers (their step-size controller branches on dual numbers),
    so we force a fixed-step Euler integration; ``n_steps`` overrides the
    solver's default ``train_solver_steps``.
    """
    z_b, was_unbatched = _ensure_batched(z)
    v_b, _ = _ensure_batched(v)

    steps = int(n_steps if n_steps is not None else spatial_ode.train_solver_steps)
    dt = 1.0 / steps

    def fn(zz):
        x = zz
        for k in range(steps):
            tau_k = torch.tensor(k * dt, dtype=x.dtype, device=x.device)
            x = x + spatial_ode.velocity_net(tau_k, x) * dt
        return x

    try:
        from torch.func import jvp
    except ImportError:
        from functorch import jvp

    _, jv = jvp(fn, (z_b,), (v_b,))
    return jv.squeeze(0) if was_unbatched else jv


def step_obs_space_explicit_jacobian(spatial_ode, dynamics, y, z, dt, noise=False):
    r"""
    One Euler--Maruyama step in observation space using the *explicit* Jacobian.

    ``y`` and ``z`` are kept in lock-step: ``z`` is the latent state, ``y`` is
    the corresponding workspace state. After the step we update ``z`` by the
    latent Euler step and recompute ``y`` via the Jacobian-pushed increment;
    callers can periodically re-anchor by ``y = decode(z)`` if drift accumulates.
    """
    f = dynamics.velocity(z)
    J = decode_jacobian(spatial_ode, z)

    if J.dim() == 2:
        Jf = J @ f.squeeze(0)
        dy = Jf * dt
        if noise:
            K = dynamics.K
            dB = torch.randn(K.shape[1], device=z.device, dtype=z.dtype) * (dt ** 0.5)
            dy = dy + (J @ K @ dB)
        y_next = y + dy
    else:
        Jf = torch.bmm(J, f.unsqueeze(-1)).squeeze(-1)
        dy = Jf * dt
        if noise:
            K = dynamics.K
            B = z.shape[0]
            dB = torch.randn(B, K.shape[1], device=z.device, dtype=z.dtype) * (dt ** 0.5)
            dy = dy + torch.bmm(J, (K @ dB.T).T.unsqueeze(-1)).squeeze(-1)
        y_next = y + dy

    z_next = z + f * dt
    return y_next, z_next


def step_obs_space_jvp(spatial_ode, dynamics, y, z, dt, noise=False):
    r"""
    One Euler--Maruyama step using *matrix-free* JVPs.

    Drift contribution: ``J f``; diffusion: ``J K dB``. Each is computed as a
    single JVP, so cost is :math:`\mathcal{O}(\text{NFE})` rather than
    :math:`\mathcal{O}(d \cdot \text{NFE})`.
    """
    f = dynamics.velocity(z)
    Jf = decode_jvp(spatial_ode, z, f)
    dy = Jf * dt

    if noise:
        K = dynamics.K
        if z.dim() == 1:
            dB = torch.randn(K.shape[1], device=z.device, dtype=z.dtype) * (dt ** 0.5)
            v = K @ dB
        else:
            dB = torch.randn(z.shape[0], K.shape[1], device=z.device, dtype=z.dtype) * (dt ** 0.5)
            v = (K @ dB.T).T
        Jv = decode_jvp(spatial_ode, z, v)
        dy = dy + Jv

    z_next = z + f * dt
    y_next = y + dy
    return y_next, z_next
