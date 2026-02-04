
import gymnasium as gym
import torch
import torch.distributions as dist
import numpy as np
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt



class PPO_Net(torch.nn.Module):
    def __init__(self, n_observations,n_actions):
        super(PPO_Net, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(n_observations, 256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, 256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, n_actions) 
        )
        self.log_std = torch.nn.Parameter(torch.zeros(n_actions))
    
    def forward(self, x):
        mean = self.fc(x)
        std = self.log_std.exp()
        return mean, std

    def get_action(self, state):
        mean, std = self.forward(state)
        normal_dist = dist.Normal(mean, std)
        action = normal_dist.sample()
        log_prob = normal_dist.log_prob(action).sum(dim=-1)  # Sum over action dimensions
        return action, log_prob

    def get_log_prob(self,state,action):
        mean, std = self.forward(state)
        normal_dist = dist.Normal(mean, std)
        log_prob = normal_dist.log_prob(action).sum(dim=-1)
        return log_prob



class Value_Net(torch.nn.Module):
    def __init__(self, n_observations):
        super(Value_Net, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(n_observations, 256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, 256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, 1) 
        )
    
    def forward(self, x):
        state_value = self.fc(x)
        return state_value

    

def get_advantage_vectorized(state_vals, rewards, dones, gamma=0.99, lam=0.95):
    """
    Vectorized GAE computation across multiple environments.
    
    Args:
        state_vals: [T, num_envs]
        rewards:    [T, num_envs]
        dones:      [T, num_envs]
    
    Returns:
        advantages: [T, num_envs]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T, num_envs = rewards.shape
    advantages = torch.zeros_like(rewards).to(device)
    last_gae = torch.zeros(num_envs).to(device)
    
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = torch.zeros(num_envs).to(device)
        else:
            next_value = state_vals[t + 1]
        
        # Mask out next_value where episode ended
        next_value = next_value * (1 - dones[t])
        
        delta = rewards[t] + gamma * next_value - state_vals[t]
        
        # Reset last_gae where episodes ended
        last_gae = last_gae * (1 - dones[t])
        
        advantages[t] = delta + gamma * lam * last_gae #[T, num_envs]
        last_gae = advantages[t]
    
    return advantages


def update_model(policy,value_net,states, actions, rewards,dones, log_probs,policy_opt,value_opt,batch_size = 2, epochs = 2):
    """
    Does mini-batch SGD using collected 
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #have collected state, action, etc lists get converted into tensors here and make a dataloader for them
    eps = .2
    states     = torch.tensor(np.array(states), dtype=torch.float).to(device)
    actions    = torch.tensor(np.array(actions), dtype=torch.float).to(device)
    rewards    = torch.tensor(np.array(rewards), dtype=torch.float).to(device)
    dones      = torch.tensor(np.array(dones), dtype=torch.float).to(device)
    log_probs = torch.stack(log_probs).to(device)



    T, num_envs = rewards.shape
    obs_dim = states.shape[-1]
    act_dim = actions.shape[-1]

    with torch.no_grad():
        # Value net expects [batch, obs_dim], so reshape for forward pass
        state_values = value_net(states.reshape(-1, obs_dim)).squeeze(-1)   # [T * num_envs]
        state_values = state_values.reshape(T, num_envs)                    # [T, num_envs]

        advantages = get_advantage_vectorized(state_values,rewards,dones)   # [T, num_envs]

        returns = advantages + state_values #do this before normalizing advantages

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) 


    
    #flatten everything, everything should be aligned across all actors now, so dataloaders can work just fine
    states     = states.reshape(-1, obs_dim)      # [T * num_envs, obs_dim]
    actions    = actions.reshape(-1, act_dim)     # [T * num_envs, act_dim]
    log_probs  = log_probs.reshape(-1)            # [T * num_envs]
    advantages = advantages.reshape(-1)           # [T * num_envs]
    returns    = returns.reshape(-1)              # [T * num_envs]




    dataset = TensorDataset(states, actions, log_probs, advantages, returns)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    #------------------------------------------------------------------------------

    for i in range(epochs):
        for batch_states, batch_actions, batch_logprob_old, batch_adv, batch_returns in loader:
            policy_opt.zero_grad()
            value_opt.zero_grad()

            


            new_log_probs = policy.get_log_prob(batch_states,batch_actions)
            state_vals = value_net(batch_states).squeeze(-1) #[T,]

            ratio = torch.exp(new_log_probs -batch_logprob_old)

            adv = batch_adv.squeeze(-1) #to collapse it back to just [T,]
            unclipped = ratio * adv
            clipped = torch.clip(ratio, 1 - eps, 1 + eps) * adv
            clip_loss = -torch.min(unclipped,clipped).mean()


            b_returns = batch_returns
            val_loss = F.mse_loss(state_vals,b_returns)

            clip_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            policy_opt.step()

            val_loss.backward()
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=0.5)
            value_opt.step()



def eval_policy(env_name,policy_network,eval_num):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make(env_name,render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        episode_trigger = lambda num: num % 1 == 0,
        video_folder="saved-video-folder",
        name_prefix=f"video-{eval_num}",
    )

    episode_over = False
    total_reward = 0

    obs, _ = env.reset()


    while not episode_over:
        with torch.no_grad():
                mean, _ = policy_network.forward(torch.tensor(obs, dtype=torch.float).to(device))
                action = torch.clip(mean,-1,1).detach().cpu().numpy()

                obs, reward, terminated, truncated, _ = env.step(action)

                episode_over = terminated or truncated

                total_reward += reward


    env.close()
    return total_reward

        








def train_PPO_agent(env_name,actors=4):

    #render_mode="human"
    envs = gym.make_vec(env_name, num_envs=actors, vectorization_mode="sync")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    action_space = envs.action_space.shape[1]
    observation_space = envs.observation_space.shape[1] # gets correct amount for each env

    print(action_space, observation_space)


    Policy_network = PPO_Net(observation_space,action_space).to(device)
    Value_network = Value_Net(observation_space).to(device)

    #paper doesn't say which lr to use for critic.
    policy_opt = torch.optim.Adam(Policy_network.parameters(), lr=3e-4)
    value_opt  = torch.optim.Adam(Value_network.parameters(),  lr=1e-3)

    epochs = 10
    batch_size = 64

    total_timesteps = 1000000
    horizon = 2048 

    eval_time = 50000
    eval_steps = 0

    total_rewards = []
    eval_num = 1

    step = 0 

    pbar = tqdm(total=total_timesteps)

    while step <= total_timesteps:

        all_states, all_actions, all_rewards,all_dones, all_log_probs = [], [], [], [], []

        obs, _ = envs.reset()


        for i in range(horizon):  

            with torch.no_grad():
                actions, log_probs = Policy_network.get_action(torch.tensor(obs, dtype=torch.float).to(device))
                log_probs = log_probs.detach().cpu()
            actions = torch.clip(actions,-1,1).detach().cpu().numpy()

            next_obs, rewards, terminated, truncated, _ = envs.step(actions)

            dones = np.logical_or(terminated, truncated)
            
            all_states.append(obs)
            all_actions.append(actions)
            all_rewards.append(rewards)
            all_dones.append(dones)
            all_log_probs.append(log_probs)

            step += actors #samples collected 
            eval_steps += 1
            pbar.update(actors)


            #wrapper handles env resetting
            obs = next_obs

        update_model(Policy_network,Value_network,all_states,all_actions,all_rewards,all_dones,all_log_probs,policy_opt,value_opt,batch_size,epochs)

        if eval_steps >= eval_time:
            print(f"Starting eval {eval_num} at step: {step}")
            eval_steps = 0
            total_rewards.append(eval_policy(env_name,Policy_network,eval_num))
            eval_num += 1

    pbar.close()
    #final eval
    total_rewards.append(eval_policy(env_name,Policy_network,eval_num))
    envs.close()

    return total_rewards




rewards = train_PPO_agent("Walker2d-v5")


episodes = np.arange(1, len(rewards) + 1)
plt.plot(episodes,rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Rewards vs Episodes')
plt.grid(True)
plt.show()