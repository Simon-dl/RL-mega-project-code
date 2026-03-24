import h5py
import numpy as np
import sys
from torch.utils.data import  TensorDataset, DataLoader
import torch

# Open the file in read mode
def explore_dataset(hdf5_path):
    with h5py.File('Diffusion_Policy/dataset/low_dim_v15.hdf5', 'r') as f:
        # Explore the structure
        print("Top-level keys:", list(f.keys()))
        
        # Robomimic datasets typically have a 'data' group
        if 'data' in f:
            data_group = f['data']
            print("\nEpisodes in dataset:", len(data_group.keys()))
            
            # Each episode is typically named 'demo_0', 'demo_1', etc.
            episode_key = list(data_group.keys())[0]  # Get first episode
            episode = data_group[episode_key]
            
            print(f"\nKeys in {episode_key}:", list(episode.keys()))
            
            # Episode keys:
            # - 'actions': action sequences
            # - 'obs': observations (may have subkeys like 'robot0_eef_pos', etc.)
            # - 'next_obs': next observations
            # - 'states': states
            # - 'rewards': reward signals
            # - 'dones': episode termination flags
            
            if 'actions' in episode:
                actions = episode['actions'][:]
                print(f"Actions shape: {actions.shape}")
                print(f"Actions: {actions[0]}")
            
            if 'obs' in episode:
                total_shape = 0
                obs_group = episode['obs']
                print(f"\n Observation keys: {list(obs_group.keys())}")
                for key in obs_group.keys():
                    print(f"{key}: {obs_group[key][0].shape}")
                    total_shape += obs_group[key][0].shape[0]
                print("total shape: ", total_shape)

            if 'next_obs' in episode:
                next_obs_group = episode['next_obs']
                print(f"\n Next observation keys: {list(next_obs_group.keys())}")
            

            if 'states' in episode:
                    states = episode['states'][:]
                    print(f"\n States shape: {states.shape}")
                    print(f"States: {states[0]}")



def compute_state_stats(state_sequences, eps=1e-6):
    """
    state_sequences: tensor of shape (N, obs_horizon, state_dim)
    returns:
        state_mean: (state_dim,)
        state_std:  (state_dim,)
    """
    flat_states = state_sequences.reshape(-1, state_sequences.shape[-1])
    state_mean = flat_states.mean(dim=0)
    state_std = flat_states.std(dim=0, unbiased=False)
    state_std = torch.clamp(state_std, min=eps)
    return state_mean, state_std

def normalize_states(x, state_mean, state_std):
    """
    x: (..., state_dim)
    state_mean/state_std: (state_dim,)
    """
    return (x - state_mean) / state_std


def normalize_actions(x, a_min, a_max):
    """Scale each dimension independently to [-1, 1]"""
    range_vals = a_max - a_min
    # Handle constant dimensions (like gripper stuck at -1.0)
    # Shift to zero-mean without scaling, as the paper recommends
    constant_mask = range_vals < 1e-6
    
    normalized = torch.zeros_like(x)
    # Normal dimensions: scale to [-1, 1]
    normalized[..., ~constant_mask] = (
        2 * (x[..., ~constant_mask] - a_min[~constant_mask]) 
        / range_vals[~constant_mask] - 1
    )
    # Constant dimensions: just center at zero
    normalized[..., constant_mask] = x[..., constant_mask] - x[..., constant_mask].mean()
    
    return normalized

def denormalize_actions(x, a_min, a_max):
    """Scale from [-1, 1] back to original range
    
    Args:
        x: The normalized actions. tensor of shape (trajectory_len, action_shape)
        a_min: The minimum values of the actions.
        a_max: The maximum values of the actions.
    Returns:
        denormalized: The denormalized actions. numpy array
    """
    range_vals = a_max - a_min
    constant_mask = range_vals < 1e-6
    
    denormalized = torch.zeros_like(x)
    denormalized[..., ~constant_mask] = (
        (x[..., ~constant_mask] + 1) / 2 * range_vals[~constant_mask] 
        + a_min[~constant_mask]
    )
    # Constant dimensions: restore original value
    denormalized[..., constant_mask] = a_min[constant_mask]
    
    return denormalized.numpy()


def concat_observation_keys(obs_group):
    """
    Concatenates the observation keys into a single state.
    Args:
        obs_group: The observation group.
    Returns:
        state: The concatenated state.
    """
    obs_keys = [
    "object",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_eef_quat_site",
    "robot0_gripper_qpos",
    "robot0_gripper_qvel",
    "robot0_joint_pos",
    "robot0_joint_pos_cos",
    "robot0_joint_pos_sin",
    "robot0_joint_vel",
    ]
    state = np.concatenate([obs_group[key][:] for key in obs_keys], axis=1) #concate along feature axis
    return state

def load_and_process_dataset(hdf5_path, obs_horizon=2, pred_horizon=10):
    """
    Loads and processes the dataset from the hdf5 file.
    Args:
        hdf5_path: Path to the hdf5 file.
        obs_horizon: The number of steps to look back for the observation.
        pred_horizon: The number of steps to look forward for the action.
    Returns:
        dataset: A TensorDataset containing the action and state sequences.
        action_min: The minimum values of the actions.
        action_max: The maximum values of the actions.

    action max and min are used to normalize the actions.
    """
    action_sequences = []
    state_sequences = []
    
    with h5py.File(hdf5_path, 'r') as f:
        data_group = f['data']
        for episode_key in data_group.keys():
            episode = data_group[episode_key]
            actions = episode['actions'][:]
            obs = episode['obs']
            states = concat_observation_keys(obs)
            ep_len = len(actions)
            
            # Sliding window, not chunking
            for t in range(ep_len - pred_horizon + 1):
                # Observation: T_o steps ending at t
                obs_start = max(0, t - obs_horizon + 1)
                obs_window = states[obs_start:t + 1]
                
                # Pad if we're near the start of the episode
                if len(obs_window) < obs_horizon:
                    pad = np.repeat(obs_window[:1], obs_horizon - len(obs_window), axis=0)
                    obs_window = np.concatenate([pad, obs_window], axis=0)
                
                # Action: T_p steps starting at t
                act_window = actions[t:t + pred_horizon]
                
                state_sequences.append(obs_window)      # (T_o, state_dim), 53 dim
                action_sequences.append(act_window)      # (T_p, action_dim)
                
    action_sequences = np.array(action_sequences)
    state_sequences = np.array(state_sequences)
    
    action_sequences = torch.tensor(action_sequences,dtype=torch.float)
    state_sequences = torch.tensor(state_sequences,dtype=torch.float)
    
    state_mean, state_std = compute_state_stats(state_sequences)
    print("total dataset size:  ")
    print("action_sequences shape: ", action_sequences.shape)
    print("state_sequences shape: ", state_sequences.shape)
    

    all_actions = action_sequences.reshape(-1, action_sequences.shape[-1])  # (N*10, 7)
    action_min = all_actions.min(dim=0).values  # (7,)
    action_max = all_actions.max(dim=0).values  # (7,)
    print("action_min: ", action_min)
    print("action_max: ", action_max)

    print("--------------------------------------------------------------------------------\n")


    dataset = TensorDataset(
        action_sequences,
        state_sequences
    )

    return dataset, action_min, action_max, state_mean, state_std

 




#explore_dataset('Diffusion_Policy/dataset/low_dim_v15.hdf5')
#load_and_process_dataset('Diffusion_Policy/dataset/low_dim_v15.hdf5')