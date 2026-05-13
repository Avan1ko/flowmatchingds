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
nr_epochs = 1000
ckpt_every = 25
## filename ##
filename = 'Angle'

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

def compute_final_metrics(expert_trajectories, model, device, training_time, threshold=0.1):
    mse_list, dtw_list, jerk_list = [], [], []
    success_count = 0
    
    def simple_dtw(s1, s2):
        n, m = len(s1), len(s2)
        cost_mat = np.zeros((n + 1, m + 1))
        cost_mat[1:, 0], cost_mat[0, 1:] = np.inf, np.inf
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                dist = np.linalg.norm(s1[i-1] - s2[j-1])
                cost_mat[i, j] = dist + min(cost_mat[i-1, j], cost_mat[i, j-1], cost_mat[i-1, j-1])
        return cost_mat[n, m]

    print("\n" + "="*30)
    print("FINAL EVALUATION METRICS")
    print("="*30)
    
    start_inf = time.time()
    for expert in expert_trajectories:
        expert_np = np.asarray(expert)
        y0 = to_torch(expert_np[0:1], device)
        
        with torch.no_grad():
            # Standard generate_trj usually returns a full path
            gen_trj = model.generate_trj(y0, T=len(expert_np))
            gen_trj = to_numpy(gen_trj).squeeze()
        
        # RMSE
        mse = np.sqrt(np.mean(np.linalg.norm(expert_np - gen_trj, axis=1)**2))
        mse_list.append(mse)
        
        # DTW
        dtw_list.append(simple_dtw(expert_np, gen_trj))
        
        # Success (Reaching the expert's goal)
        final_dist = np.linalg.norm(gen_trj[-1] - expert_np[-1])
        if final_dist < threshold:
            success_count += 1
            
        # Smoothness: Mean Jerk
        # Calculated via finite difference: jerk ≈ Δ³x / Δt³
        jerk = np.diff(gen_trj, n=3, axis=0)
        jerk_list.append(np.mean(np.linalg.norm(jerk, axis=1)))

    inf_time = (time.time() - start_inf) / len(expert_trajectories)

    results = {
        "Total Train Time (s)": training_time,
        "Inference Time/Trj (s)": inf_time,
        "Avg RMSE": np.mean(mse_list),
        "Avg DTW": np.mean(dtw_list),
        "Success Rate (%)": (success_count / len(expert_trajectories)) * 100,
        "Mean Jerk (Smoothness)": np.mean(jerk_list)
    }
    
    for k, v in results.items():
        print(f"{k:25}: {v:.4f}")
    return results

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

    total_train_time = time.time() - start_train_time
    print("\nTraining complete. Running final evaluation...")
    iflow.eval()
    final_metrics = compute_final_metrics(
        data.train_data, 
        iflow, 
        device, 
        total_train_time
    )







