import h5py
import json
import robosuite

import numpy as np
from robosuite.wrappers import DataCollectionWrapper
import time
import os
from glob import glob
import imageio
import torch
from diffusion_model import DiffusionPolicyTransformer
from tqdm import tqdm

from dataset_loading import denormalize_actions, normalize_states

def sch_func(t, total_steps):
    """
    Compute the schedule function for the diffusion process.
    Args:
        t: the current timestep
        total_steps: the total number of steps in the diffusion process
    Returns:
        the schedule function
    """
    s = 0.008
    t = t.float() if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.float)
    
    def f(t_val):
        return torch.cos(((t_val / total_steps) + s) / (1 + s) * (torch.pi / 2)) ** 2
    
    return f(t) / f(torch.zeros_like(t))


def ddim_update_step(noisy_actions, alpha_bar_t, alpha_bar_tm1, eps_hat):
    """
    no more step variation with eta=0. This is a deterministic update step. 
    """
    x0_hat = (noisy_actions - torch.sqrt(1 - alpha_bar_t) * eps_hat) / torch.sqrt(alpha_bar_t)
    x0_hat = torch.clamp(x0_hat, -1, 1)  # stability clamp

    dir_coeff = torch.sqrt(torch.clamp(1 - alpha_bar_tm1, min=0))
    x_tm1 = torch.sqrt(alpha_bar_tm1) * x0_hat + dir_coeff * eps_hat

    return x_tm1

def diffusion_policy_sampler(model,state_mean,state_std,trajectory_len,actions_shape,states,train_steps, inference_steps=100):
    """
    Samples a trajectory of actions from the diffusion policy.
    Args:
        model: the diffusion policy model
        trajectory_len: the length of the trajectory, t_p in the paper, 16 in paper. hardcoded here
        actions_shape: the shape of the actions
        states: the states, a list of states
        train_steps: the number of training steps
        inference_steps: the number of inference steps
    Returns:
        actions: the actions, a tensor of shape (trajectory_len, actions_shape)
    """
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create evenly spaced timesteps from T down to 0
    # e.g., with 100 inference steps and 1000 training steps:
    # [1000, 990, 980, ..., 10, 0]
    step_size = train_steps // inference_steps
    timesteps = list(range(train_steps, 0, -step_size))

    states = torch.tensor(np.array(states), dtype=torch.float)
    states = states.unsqueeze(0).to(device) #(1, trajectory_len, 53)
    states = normalize_states(states, state_mean.to(device), state_std.to(device))
    
    #pure noise
    noisy_actions = torch.randn(1,trajectory_len,actions_shape).to(device)  

    with torch.no_grad():
        for i in range(inference_steps):

            t = timesteps[i]
            t_tensor = torch.tensor([t], dtype=torch.float).to(device)
        
            alpha_bar_t = sch_func(t, train_steps).to(device)

            if i + 1 < len(timesteps):
                t_prev = timesteps[i + 1]
            else:
                t_prev = 0

            alpha_bar_tm1 = sch_func(t_prev, train_steps).to(device)

            eps_hat = model(noisy_actions,states,t_tensor).to(device)

            # DDIM update with eta=0 for deterministic sampling
            noisy_actions = ddim_update_step(noisy_actions, alpha_bar_t, alpha_bar_tm1, eps_hat)

    actions = noisy_actions.detach().cpu()
        
    return actions.squeeze(0) #(trajectory_len, actions_shape)


def concat_observation_keys(obs_group):
    """
    Concatenates the observation keys into a single state.
    Args:
        obs_group: The observation group.
    Returns:
        state: The concatenated state.
    """
    obs_keys = [
    "object-state",
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
    state = np.concatenate([obs_group[key][:] for key in obs_keys], axis=0) #concate along feature axis
    return state

def collect_trajectory(env,model,action_min,action_max,state_mean,state_std,resample_step=8, timesteps=1000, max_fr=None):
    """Run a random policy to collect trajectories.

    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env (MujocoEnv): environment instance to collect trajectories from
        timesteps(int): how many environment timesteps to run for a given trajectory
        resample_step(int): how many steps to take before resampling the actions, T_a in the paper, 8 in paper
        action_min(tensor): the minimum values of the actions
        action_max(tensor): the maximum values of the actions
        max_fr (int): if specified, pause the simulation whenever simulation runs faster than max_fr
    
    Returns:
        str: Path to the created episode directory
    """
    # Get the data directory from the wrapper (if accessible)
    # Otherwise, we'll find it by looking for the most recent directory
    data_dir = getattr(env, 'directory', None)
    if data_dir is None:
        # Try to get it from the wrapper's attributes
        data_dir = getattr(env, 'data_directory', 'Diffusion_Policy/videos')
    
    # Store directories before reset to find the new one
    if os.path.exists(data_dir):
        dirs_before = set(os.listdir(data_dir)) if os.path.isdir(data_dir) else set()
    else:
        dirs_before = set()
    
    print("--------------------------------------------------------------------------------- \n")
    print("collecting trajectory...")
    trajectory_len = 10
    
    obs = env.reset()
    dof = env.action_dim #7

    state = concat_observation_keys(obs)

    states = [state, state] #initial states to sample the actions from
    
    actions = diffusion_policy_sampler(model,state_mean,state_std,trajectory_len,dof,states,999) #999 to avoid nan in sch_func

    actions = denormalize_actions(actions, action_min, action_max)
    action_counter = 0


    for t in range(timesteps):
        start = time.time()
        
        action = actions[action_counter]
        action_counter += 1

        obs, reward, done, _ = env.step(action)

        state = concat_observation_keys(obs)
        states.append(state)
        states.pop(0)

        if action_counter == resample_step: #receeding horizon, resample the actions
            actions = diffusion_policy_sampler(model,state_mean,state_std,trajectory_len,dof,states,999)
            actions = denormalize_actions(actions, action_min, action_max)
            action_counter = 0

        if t % 100 == 0:
            print(f"{t} steps completed on trajectory collection")

        # limit frame rate if necessary
        if max_fr is not None:
            elapsed = time.time() - start
            diff = 1 / max_fr - elapsed
            if diff > 0:
                time.sleep(diff)
    
    # Find the newly created episode directory
    if os.path.exists(data_dir):
        dirs_after = set(os.listdir(data_dir)) if os.path.isdir(data_dir) else set()
        new_dirs = dirs_after - dirs_before
        
        if new_dirs:
            # Get the most recently created directory (in case multiple were created)
            episode_dir = max(
                [os.path.join(data_dir, d) for d in new_dirs],
                key=lambda p: os.path.getctime(p) if os.path.exists(p) else 0
            )
            return episode_dir
    
    # Fallback: try to access from wrapper attribute
    if hasattr(env, 'episode_dir'):
        return env.episode_dir
    
    return None

def save_episode_video(env, ep_dir, out_path, fps=30, width=640, height=480, camera_name="frontview"):
    xml_path = os.path.join(ep_dir, "model.xml")
    with open(xml_path, "r") as f:
        env.reset_from_xml_string(f.read())

    state_paths = os.path.join(ep_dir, "state_*.npz")
    writer = imageio.get_writer(out_path, fps=fps)

    for state_file in sorted(glob(state_paths)):
        dic = np.load(state_file)
        states = dic["states"]
        for state in states:
            env.sim.set_state_from_flattened(state)
            env.sim.forward()

            # render offscreen from the simulator
            frame = env.sim.render(
                width=width,
                height=height,
                camera_name=camera_name,
            )
            # robosuite often returns images upside-down; flip if needed:
            frame = frame[::-1]

            writer.append_data(frame)

    writer.close()

def get_env_metadata(hdf5_path):
    with h5py.File(hdf5_path, 'r') as f:
        return json.loads(f["data"].attrs["env_args"])

def visualize_policy_state_based(hdf5_path,model,action_min,action_max,state_mean,state_std):
    """ takes in dataset and trained model and has it run in the environment and saves the video of the trajectory"""

    env_meta = get_env_metadata(hdf5_path)

    # First copy the kwargs and override inside the dict (so no duplicate args)
    env_kwargs = env_meta["env_kwargs"].copy()

    # OVERRIDES for visualization / offscreen rendering
    env_kwargs.update({
        "has_renderer": False,          # no onscreen window needed for rgb_array
        "has_offscreen_renderer": True, # REQUIRED for mode="rgb_array"
        "render_camera": "frontview",   # or "agentview", etc.
        "use_camera_obs": False,        # if you only need low-dim obs
        "ignore_done": True,            # optional
    })

    env = robosuite.make(
        env_name=env_meta["env_name"],
        **env_kwargs
    )

    data_directory = 'Diffusion_Policy/videos'
    env = DataCollectionWrapper(env, data_directory)

# 4. Collect trajectory
    episode_dir = collect_trajectory(env, model, action_min, action_max, state_mean, state_std, timesteps=300, max_fr=60) 

    if episode_dir:
        print(f"\nCollected episode saved to: {episode_dir}")
        
        # 5. Save episode video
        video_path = os.path.join(os.path.dirname(episode_dir), os.path.basename(episode_dir) + ".mp4")
        save_episode_video(
            env,
            episode_dir,
            video_path,
            fps=30,
        )
        print(f"Video saved to: {video_path}")
    else:
        print("Warning: Could not determine episode directory path")


    env.close()


# visualize_policy_state_based('Diffusion_Policy/dataset/low_dim_v15.hdf5',None)

# loading the model and dataset to visualize the policy for n evaluations
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ckpt = torch.load("Diffusion_Policy/model.pth", map_location=device)
# model = DiffusionPolicyTransformer(max_seq_len=16,action_dim=7,obs_dim=53,n_heads=4, d_model=256,dropout=0,blocks=10).to(device)
# model.load_state_dict(ckpt["ema_state_dict"])
# model.eval()

# print("model loaded")

# for i in range(10):
#     visualize_policy_state_based('Diffusion_Policy/dataset/tool_low_dim_v15.hdf5',model,ckpt["action_min"].to("cpu"),ckpt["action_max"].to("cpu"),ckpt["state_mean"].to(device),ckpt["state_std"].to(device))