"""
Loss functions and training step helpers for the decoupled FM + stable SDE
pipeline.

Phase A
-------
``loss_fm``         -- simulation-free Conditional Flow Matching MSE.
``loss_temporal_reg`` -- temporal regularization via inverse-ODE encode of
                        ordered windows + dummy linear predictor.
``phase_a_step``    -- combines both with the schedule described in the plan
                        (FM every step; Reg every ``reg_every_k`` steps with
                        weight ``lambda_reg``).

Phase C
-------
``loss_phase_c``    -- MSE between empirical latent increments and the linear
                        prediction ``A z * dt`` with ``A = S - D``.
"""
import torch


def loss_fm(spatial_ode, x1, base_std=1.0):
    r"""
    Simulation-free OT-style Conditional Flow Matching loss.

    Samples ``x0 ~ N(0, base_std^2 I)``, ``tau ~ U(0, 1)`` and compares
    ``v_theta((1-tau)x0 + tau x1, tau)`` against the straight-line target
    ``x1 - x0``.
    """
    batch_size = x1.shape[0]
    dim = x1.shape[1]
    device = x1.device
    dtype = x1.dtype

    x0 = torch.randn(batch_size, dim, device=device, dtype=dtype) * base_std
    tau = torch.rand(batch_size, 1, device=device, dtype=dtype)

    x_tau = (1.0 - tau) * x0 + tau * x1
    target = x1 - x0
    pred = spatial_ode.velocity_net(tau, x_tau)
    return ((pred - target) ** 2).mean()


def encode_window(spatial_ode, windows, training=True):
    r"""
    Push a batch of ordered windows ``(B, L, dim)`` through the inverse ODE and
    return the latent windows in the same shape.
    """
    B, L, D = windows.shape
    flat = windows.reshape(B * L, D)
    z_flat = spatial_ode.encode(flat, training=training)
    return z_flat.reshape(B, L, D)


def loss_temporal_reg(spatial_ode, dummy_predictor, windows, dt, training=True):
    r"""
    Temporal regularization loss

    .. math::
        \mathcal{L}_{\text{Reg}} = \mathrm{MSE}(z_{t+1} - z_t,\ \hat{A} z_t\, \Delta t)

    averaged over all valid pairs ``(t, t+1)`` in every window.
    """
    z = encode_window(spatial_ode, windows, training=training)
    z_t = z[:, :-1, :]
    dz = z[:, 1:, :] - z_t
    pred = dummy_predictor.predict_increment(z_t.reshape(-1, z_t.shape[-1]), dt)
    target = dz.reshape(-1, dz.shape[-1])
    return ((pred - target) ** 2).mean()


def phase_a_step(
    spatial_ode,
    dummy_predictor,
    optimizer,
    fm_sampler,
    reg_sampler,
    fm_batch_size,
    reg_batch_size,
    dt,
    lambda_reg,
    step_idx,
    reg_every_k=1,
    base_std=1.0,
):
    r"""
    Single optimizer step of :math:`\mathcal{L}_{\text{FM}} + \lambda \mathcal{L}_{\text{Reg}}`.

    To preserve the simulation-free benefit of CFM, the ODE-inverse pass behind
    :math:`\mathcal{L}_{\text{Reg}}` is only triggered every ``reg_every_k``
    steps. The remaining steps are pure MLP regressions.
    """
    optimizer.zero_grad()

    x1 = fm_sampler.sample(fm_batch_size)
    l_fm = loss_fm(spatial_ode, x1, base_std=base_std)

    l_reg_value = None
    if lambda_reg > 0.0 and reg_every_k > 0 and (step_idx % reg_every_k == 0):
        windows = reg_sampler.sample(reg_batch_size)
        l_reg = loss_temporal_reg(
            spatial_ode, dummy_predictor, windows, dt=dt, training=True
        )
        loss = l_fm + lambda_reg * l_reg
        l_reg_value = float(l_reg.detach())
    else:
        loss = l_fm

    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.detach()),
        "loss_fm": float(l_fm.detach()),
        "loss_reg": l_reg_value,
    }


def loss_phase_c(stable_sde, z_pairs, dt):
    r"""
    Phase C regression loss: ``MSE(z_{t+1} - z_t, A z_t * dt)``.

    Expects ``z_pairs`` of shape ``(N, 2, dim)`` (concatenated source / next
    pairs). Returns a scalar tensor.
    """
    z_t = z_pairs[:, 0, :]
    z_next = z_pairs[:, 1, :]
    target = z_next - z_t
    pred = stable_sde.velocity(z_t) * float(dt)
    return ((pred - target) ** 2).mean()


def build_z_pairs_from_trajectories(trajectories_z):
    r"""
    Build a ``(N, 2, dim)`` tensor of consecutive ``(z_t, z_{t+1})`` pairs from
    a list of per-trajectory latent tensors of shape ``(T_i, dim)``.
    """
    pairs = []
    for z in trajectories_z:
        pairs.append(torch.stack([z[:-1], z[1:]], dim=1))
    return torch.cat(pairs, dim=0)
