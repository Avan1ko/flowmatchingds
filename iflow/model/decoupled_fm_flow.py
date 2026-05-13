"""
Composite model wrapping the spatial Neural ODE (Phase A/B) and the stable
linear latent SDE (Phase C).

Provides a `forward / encode / decode / evolve / generate_trj` surface that
mirrors `iflow.model.ciflow.ContinuousDynamicFlow`, so the existing 2D
visualizers (`visualize_2d_generated_trj`, `visualize_latent_distribution`,
`visualize_vector_field`) work without modification.
"""
import torch
import torch.nn as nn


class DecoupledFMImitationFlow(nn.Module):
    def __init__(self, spatial_ode, dynamics, dim=2, device=None):
        super().__init__()
        self.spatial_ode = spatial_ode
        self.dynamics = dynamics
        self.dim = dim
        self.device = device

    @property
    def flow(self):
        return self.spatial_ode

    def encode(self, y):
        return self.spatial_ode.encode(y, training=self.training)

    def decode(self, z):
        return self.spatial_ode.decode(z, training=self.training)

    def forward(self, yt, context=None):
        z = self.encode(yt)
        zero_logdet = torch.zeros(z.shape[0], 1, dtype=z.dtype, device=z.device)
        return z, zero_logdet

    def generate_trj(self, y0, T=200, noise=False, reverse=False):
        z0 = self.encode(y0)
        trj_z = self.dynamics.generate_trj(z0, T=T, reverse=reverse, noise=noise)
        z_states = trj_z[:, 0, :] if trj_z.dim() == 3 else trj_z
        trj_y = self.decode(z_states)
        return trj_y

    def evolve(self, y0, T=200, noise=False, reverse=False):
        z0 = self.encode(y0)
        z1 = self.dynamics.evolve(z0, T=T, reverse=reverse, noise=noise)
        y1 = self.decode(z1)
        return y1
