import torch
import numpy as np
from iflow.utils.generic import to_numpy, to_torch
import matplotlib.pyplot as plt


def visualize_latent_distribution(val_trj, iflow, device, fig_number=1, save_path=None):
    n_trj = len(val_trj)
    dim = val_trj[0].shape[-1]
    ## Store the latent trajectory in a list ##
    val_z_trj= []
    val_mu_trj = []
    val_var_trj = []

    plt.figure(fig_number, figsize=(20,int(10*dim))).clf()
    fig, axs = plt.subplots(dim, 1, num=fig_number)
    for i in range(len(val_trj)):
        y_trj = to_torch(val_trj[i],device)
        z_trj, _ = iflow(y_trj)
        z_trj = to_numpy(z_trj)
        val_z_trj.append(z_trj)

        z0 = to_torch(z_trj[0,:],device)
        trj_mu, trj_var = iflow.dynamics.generate_trj_density(z0[None,:], T = val_z_trj[i].shape[0])
        val_mu_trj.append(to_numpy(trj_mu))
        val_var_trj.append(to_numpy(trj_var))

        for j in range(val_trj[i].shape[-1]):
            t = np.linspace(0,val_z_trj[i].shape[0], val_z_trj[i].shape[0])
            axs[j].plot(t,val_z_trj[i][:,j])
            l_trj = val_mu_trj[i][:,0,j] - np.sqrt(val_var_trj[i][:,0, j, j] )
            h_trj = val_mu_trj[i][:,0,j]  + np.sqrt(val_var_trj[i][:,0, j, j] )
            axs[j].fill_between(t,l_trj, h_trj, alpha=0.1)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
    else:
        plt.draw()
        plt.pause(0.001)


def _compute_vector_field(val_trj, iflow, device, n_grid=25, margin=0.5):
    _trajs = np.zeros((0, 2))
    for trj in val_trj:
        _trajs = np.concatenate((_trajs, trj), 0)
    lo = _trajs.min(0) - margin
    hi = _trajs.max(0) + margin

    x = np.linspace(lo[0], hi[0], n_grid)
    y = np.linspace(lo[1], hi[1], n_grid)
    xx, yy = np.meshgrid(x, y)

    hv = torch.tensor(
        np.stack([xx.ravel(), yy.ravel()], axis=1), dtype=torch.float32
    )
    if device is not None:
        hv = hv.to(device)

    hv_t1 = iflow.evolve(hv, T=3)
    hv_np = hv.detach().cpu().numpy()
    hv_t1 = hv_t1.detach().cpu().numpy()

    vel = hv_t1 - hv_np
    vel_x = vel[:, 0].reshape(n_grid, n_grid)
    vel_y = vel[:, 1].reshape(n_grid, n_grid)
    return xx, yy, vel_x, vel_y


def _plot_quiver(xx, yy, vel_x, vel_y, val_trj, fig_number):
    speed = np.sqrt(vel_x ** 2 + vel_y ** 2)
    max_speed = float(speed.max()) if speed.size else 1.0
    if max_speed < 1e-12:
        max_speed = 1.0

    nx_x = vel_x / max_speed
    nx_y = vel_y / max_speed

    fig = plt.figure(fig_number, figsize=(10, 10))
    plt.clf()
    ax = plt.gca()

    r = np.power(np.add(np.power(nx_x,2), np.power(nx_y,2)),0.5)

    q = ax.quiver(
        xx,
        yy,
        nx_x/r,
        nx_y/r,
        pivot="middle",
        angles="xy",
        scale_units="xy",
        scale=1.0 / (0.9 * (xx[0, 1] - xx[0, 0])),
        width=0.003,
    )
    # fig.colorbar(q, ax=ax, fraction=0.04, pad=0.02, label="speed")

    for i in range(len(val_trj)):
        ax.plot(val_trj[i][:, 0], val_trj[i][:, 1], "b", linewidth=1.5)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_aspect("equal", adjustable="box")
    return fig


def visualize_vector_field(
    val_trj, iflow, device, fig_number=1, save_path=None, n_grid=25
):
    xx, yy, vel_x, vel_y = _compute_vector_field(
        val_trj, iflow, device, n_grid=n_grid
    )
    fig = _plot_quiver(xx, yy, vel_x, vel_y, val_trj, fig_number)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
    else:
        plt.draw()
        plt.pause(0.05)


def save_vector_field(val_trj, iflow, device, save_fig, fig_number=1, n_grid=25):
    xx, yy, vel_x, vel_y = _compute_vector_field(
        val_trj, iflow, device, n_grid=n_grid
    )
    fig = _plot_quiver(xx, yy, vel_x, vel_y, val_trj, fig_number)
    fig.savefig(save_fig, bbox_inches="tight", dpi=150)
    plt.close(fig)
