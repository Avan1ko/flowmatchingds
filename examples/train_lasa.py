import os, sys, time
import torch
import torch.optim as optim
from iflow.dataset import lasa_dataset
from torch.utils.data import DataLoader
from iflow import model
from iflow.trainers import goto_dynamics_train
from iflow.utils import to_numpy, to_torch, makedirs
from iflow.visualization import visualize_latent_distribution, visualize_vector_field, visualize_2d_generated_trj
from iflow.test_measures import log_likelihood, iros_evaluation


percentage = .99
batch_size = 100
depth = 10
## optimization ##
lr = 0.001
weight_decay = 0.
## training variables ##
nr_epochs = 51
ckpt_every = 25
## filename ##
filename = 'LShape'

######### GPU/ CPU #############
#device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

#### Invertible Flow model #####
def main_layer(dim):
    return  model.CouplingLayer(dim)


def create_flow_seq(dim, depth):
    chain = []
    for i in range(depth):
        chain.append(main_layer(dim))
        chain.append(model.RandomPermutation(dim))
        chain.append(model.LULinear(dim))
    chain.append(main_layer(dim))
    return model.SequentialFlow(chain)

if __name__ == '__main__':
    start_train_time = time.time()
    ########## Data Loading #########
    data = lasa_dataset.LASA(filename = filename)
    dim = data.dim
    params = {'batch_size': batch_size, 'shuffle': True}
    dataloader = DataLoader(data.dataset, **params)
    ######### Model #########
    dynamics = model.TanhStochasticDynamics(dim, dt=0.01, T_to_stable=2.5)
    #dynamics = model.LinearStochasticDynamics(dim, dt=0.01, T_to_stable=2.5)
    flow = create_flow_seq(dim, depth)
    iflow = model.ContinuousDynamicFlow(dynamics=dynamics, model=flow, dim=dim).to(device)
    ########## Optimization ################
    params = list(flow.parameters()) + list(dynamics.parameters())
    optimizer = optim.Adamax(params, lr = lr, weight_decay= weight_decay)
    plot_dir = os.path.join(os.path.dirname(__file__), 'lasa_plots', filename)
    makedirs(plot_dir)
    weights_dir = os.path.join(os.path.dirname(__file__), 'lasa_weights')
    makedirs(weights_dir)
    #######################################
    for i in range(nr_epochs):
        ## Training ##
        for local_x, local_y in dataloader:
            dataloader.dataset.set_step()
            optimizer.zero_grad()
            loss = goto_dynamics_train(iflow, local_x, local_y)
            loss.backward(retain_graph=True)
            optimizer.step()

        ## Validation ##
        if i % ckpt_every == 0:
            with torch.no_grad():
                iflow.eval()

                visualize_2d_generated_trj(
                    data.train_data, iflow, device, fig_number=2,
                    save_path=os.path.join(plot_dir, 'epoch_{:04d}_traj2d.png'.format(i)))
                visualize_latent_distribution(
                    data.train_data, iflow, device, fig_number=1,
                    save_path=os.path.join(plot_dir, 'epoch_{:04d}_latent.png'.format(i)))
                visualize_vector_field(
                    data.train_data, iflow, device, fig_number=3,
                    save_path=os.path.join(plot_dir, 'epoch_{:04d}_vector_field.png'.format(i)))
                iros_evaluation(data.train_data, iflow, device)

                ## Prepare Data ##
                step = 20
                trj = data.train_data[0]
                trj_x0 = to_torch(trj[:-step,:], device)
                trj_x1 = to_torch(trj[step:,:], device)
                log_likelihood(trj_x0, trj_x1, step, iflow, device)
                print('The Variance of the latent dynamics are: {}'.format(torch.exp(iflow.dynamics.log_var)))
                print('The Velocity of the latent dynamics are: {}'.format(iflow.dynamics.Kv[0,0]))

            ckpt_path = os.path.join(
                weights_dir,
                '{}_train{:04d}epochs_epoch{:04d}.pt'.format(filename, nr_epochs, i),
            )
            torch.save(
                {
                    'shape': filename,
                    'epoch': i,
                    'nr_epochs': nr_epochs,
                    'iflow_state_dict': iflow.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'dim': dim,
                    'depth': depth,
                    'lr': lr,
                    'weight_decay': weight_decay,
                },
                ckpt_path,
            )
            print('Saved checkpoint: {}'.format(ckpt_path))

    last_epoch = nr_epochs - 1
    if last_epoch % ckpt_every != 0:
        ckpt_path = os.path.join(
            weights_dir,
            '{}_train{:04d}epochs_epoch{:04d}.pt'.format(
                filename, nr_epochs, last_epoch),
        )
        torch.save(
            {
                'shape': filename,
                'epoch': last_epoch,
                'nr_epochs': nr_epochs,
                'iflow_state_dict': iflow.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dim': dim,
                'depth': depth,
                'lr': lr,
                'weight_decay': weight_decay,
            },
            ckpt_path,
        )
        print('Saved final checkpoint: {}'.format(ckpt_path))
    print("Total Train Time")
    total_train_time = time.time() - start_train_time
    print(total_train_time)
    iros_evaluation(data.train_data, iflow, device)








