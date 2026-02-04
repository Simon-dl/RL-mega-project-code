
import gymnasium as gym
import torch
import torch.distributions as dist
import numpy as np
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import tqdm
import matplotlib.pyplot as plt



class PPO_Net(torch.nn.Module):
    def __init__(self, n_observations,n_actions):
        super(PPO_Net, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(n_observations,64),
            torch.nn.Tanh(),
            torch.nn.Linear(64,64),
            torch.nn.Tanh(),
            torch.nn.Linear(64,n_actions) 
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
            torch.nn.Linear(n_observations,64),
            torch.nn.Tanh(),
            torch.nn.Linear(64,64),
            torch.nn.Tanh(),
            torch.nn.Linear(64,1) 
        )
    
    def forward(self, x):
        state_value = self.fc(x)
        return state_value

    

def get_advantage(state_vals,rewards,dones, gamma = .99, lam = .95 ):
    #δt​=rt​+γV(st+1​)−V(st​)
    #A^T−1​=δT−1​
    #A^T−2=δT−2+(γλ)A^T−1\hat{A}_{T-2} = \delta_{T-2} + (\gamma\lambda)\hat{A}_{T-1}A^T−2​=δT−2​+(γλ)A^T−1​

    T = len(rewards) 
    advantages = torch.zeros(T)
    last_gae = 0


    for t in reversed(range(T)):
        # If episode ended or it's the last timestep, no bootstrapping
        if dones[t] or t == T - 1:
            next_value = 0
            last_gae = 0
        else:
            next_value = state_vals[t + 1]
        
        # TD error at this timestep
        delta = rewards[t] + gamma * next_value - state_vals[t]
        
        # GAE recursive formula: A_t = delta_t + (gamma * lambda) * A_{t+1}
        advantages[t] = delta + gamma * lam * last_gae
        last_gae = advantages[t]
    
    return advantages


def update_model(policy,value_net,states, actions, rewards,dones, log_probs,policy_opt,value_opt,batch_size = 2, epochs = 2):
    """
    Does mini-batch SGD using collected 
    """

    #have collected state, action, etc lists get converted into tensors here and make a dataloader for them
    eps = .2
    states     = torch.tensor(np.array(states), dtype=torch.float)
    actions    = torch.tensor(np.array(actions), dtype=torch.float)
    rewards    = torch.tensor(np.array(rewards), dtype=torch.float)
    dones      = torch.tensor(np.array(dones), dtype=torch.float)
    log_probs  = torch.tensor(np.array(log_probs), dtype=torch.float)

    with torch.no_grad():
        state_values = value_net(states).squeeze(-1) #hopefully okay to do 2048 horizon at once, unsqeeze needed for returns calc
        advantages = get_advantage(state_values,rewards,dones)

        returns = advantages + state_values #do this before normalizing advantages

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) 


    
    advantages = advantages.unsqueeze(-1)    # [T, 1], to make it match
    returns    = returns.unsqueeze(-1)       # [T, 1]

    dataset = TensorDataset(states, actions, log_probs, advantages, returns)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    #------------------------------------------------------------------------------

    for i in tqdm.tqdm(range(epochs)):
        for batch_states, batch_actions, batch_logprob_old, batch_adv, batch_returns in loader:
            policy_opt.zero_grad()
            value_opt.zero_grad()

            


            new_log_probs = policy.get_log_prob(batch_states,batch_actions)
            state_vals = value_net(batch_states)

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



def eval_policy(env_name,policy_network):

    #MAke this record videos

    env = gym.make("Hopper-v5")

    episode_over = False
    total_reward = 0

    obs, _ = env.reset()


    while not episode_over:
        with torch.no_grad():
                mean, _ = policy_network.forward(torch.tensor(obs, dtype=torch.float))
                action = torch.clip(mean,-1,1).detach().numpy()

                obs, reward, terminated, truncated, _ = env.step(action)

                episode_over = terminated or truncated

                total_reward += reward



    print(total_reward)
    return total_reward

        








def train_PPO_agent(env_name):

    #render_mode="human"
    env = gym.make(env_name)

    action_space = env.action_space.shape[0]
    observation_space = env.observation_space.shape[0]

    Policy_network = PPO_Net(observation_space,action_space)
    Value_network = Value_Net(observation_space)

    #paper doesn't say which lr to use for critic.
    policy_opt = torch.optim.Adam(Policy_network.parameters(), lr=3e-4)
    value_opt  = torch.optim.Adam(Value_network.parameters(),  lr=1e-3)

    epochs = 10
    batch_size = 32

    total_timesteps = 15000

    eval_time = 5000
    eval_steps = 0

    total_rewards = []

    step = 0 


    while step <= total_timesteps:

        states, actions, rewards,dones, log_probs = [], [], [], [], []

        obs, _ = env.reset()


        for horizon in range(2048):  

            with torch.no_grad():
                action, log_prob = Policy_network.get_action(torch.tensor(obs, dtype=torch.float))
                log_prob.detach()
            action = torch.clip(action,-1,1).detach().numpy()

            next_obs, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated
            
            states.append(obs)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob)

            step += 1
            eval_steps += 1



            if terminated or truncated:
                obs, _ = env.reset()
            else:
                obs = next_obs

        update_model(Policy_network,Value_network,states,actions,rewards,dones,log_probs,policy_opt,value_opt,batch_size,epochs)

        if eval_steps >= eval_time:
            print("here")
            eval_steps = 0
            total_rewards.append(eval_policy(env_name,Policy_network))

    #final eval
    total_rewards.append(eval_policy(env_name,Policy_network))

    return total_rewards




rewards = train_PPO_agent("Hopper-v5")


episodes = np.arange(1, len(rewards) + 1)
plt.plot(episodes,rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Rewards vs Episodes')
plt.grid(True)
plt.show()


