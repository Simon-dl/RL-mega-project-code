import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from dataset_loading import load_and_process_dataset
from diffusion_model import DiffusionPolicyTransformer
import matplotlib.pyplot as plt
from tqdm import tqdm
from eval_and_viz import visualize_policy_state_based, sch_func
from dataset_loading import normalize_actions, normalize_states



class EMA:
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = {name: param.clone().detach() 
                       for name, param in model.named_parameters()}
    
    def update(self, model):
        for name, param in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(
                param.data, alpha=1 - self.decay
            )
    
    def apply(self, model):
        """Swap EMA weights into the model for evaluation"""
        self.backup = {name: param.clone() 
                       for name, param in model.named_parameters()}
        for name, param in model.named_parameters():
            param.data.copy_(self.shadow[name])
    
    def restore(self, model):
        """Restore original weights after evaluation"""
        for name, param in model.named_parameters():
            param.data.copy_(self.backup[name])
#----------------------------------------------------------------------------------------------------------
#Training Loop
#----------------------------------------------------------------------------------------------------------



def train_diffusion_policy(dataset_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    loss_fn = nn.MSELoss(reduction='mean')
    total_steps = 1000
    obs_horizon = 2
    pred_horizon = 10

    

    model = DiffusionPolicyTransformer(max_seq_len=16,action_dim=7,obs_dim=53,n_heads=4, d_model=256,dropout=0,blocks=10) #paper uses d_model=256, blocks=8, n_heads=4?
    model.to(device)

    epochs = 1
    #Not sure what optimizer authors use, so I'm using Adam. lr 1e-4, wdecay 1e-3 for full training.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4,weight_decay=1e-3) #weight_decay=1e-3 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) #not doing warm up

    ema = EMA(model, decay=0.995)
    batch_size = 256
    mean_losses = []

    dataset, action_min, action_max, state_mean, state_std = load_and_process_dataset(dataset_path, obs_horizon, pred_horizon)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    state_mean = state_mean.to(device)
    state_std = state_std.to(device)
    

    for epoch in tqdm(range(epochs),desc="Training Diffusion Policy"):
        loss_list = []
        for batch in loader:
            optimizer.zero_grad()
            actions, states = batch #(batch_size, action_horizon, action_shape) and (batch_size, action_horizon, state_shape)
            
            states = states.to(device)

            states = normalize_states(states, state_mean, state_std) #normalize states to mean 0 and std 1

            current_batch_size = actions.shape[0] #since batch at end might not be the same size as the batch_size

            normalized_actions = normalize_actions(actions, action_min, action_max).to(device)
            
            sample_noise = torch.randn_like(normalized_actions).to(device)
        

            diff_steps = torch.randint(low=1,high=total_steps+1,size=(current_batch_size,)).to(device) #(batch_size,)

            alpha_bar_t = sch_func(diff_steps,total_steps) #(batch_size,)
            alpha_bar_t = alpha_bar_t.view(current_batch_size, 1, 1).to(device)  #(batch_size, 1, 1) so it broadcasts correctly

            #uses same DDPM way to get to arbitary timestep
            noisy_actions = ( torch.sqrt(alpha_bar_t) * normalized_actions + torch.sqrt(1 - alpha_bar_t) * sample_noise ).to(device)
            #should be(batch_size, action_horizon, action_shape)


            eps = model(noisy_actions, states, diff_steps).to(device)

            loss = loss_fn(eps, sample_noise).to(device)
            loss_list.append(loss.item())


            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ema.update(model)


        mean_loss = np.mean(loss_list)
        print("mean loss: ", mean_loss)
        mean_losses.append(mean_loss)
        scheduler.step()

    print("final loss: ", mean_losses[-1])

    torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "epoch": epoch,
    "action_min": action_min.cpu(),
    "action_max": action_max.cpu(),
    "state_mean": state_mean.cpu(),
    "state_std": state_std.cpu(),
    "ema_state_dict": ema.shadow,
    }, "Diffusion_Policy/model.pth")

    plt.plot(mean_losses)
    plt.show()


    #sample a trajectory
    ema.apply(model)
    visualize_policy_state_based(dataset_path,model,action_min,action_max,state_mean,state_std)
    ema.restore(model)



if __name__ == "__main__":
    dataset_path = 'Diffusion_Policy/dataset/low_dim_v15.hdf5'
    train_diffusion_policy(dataset_path)