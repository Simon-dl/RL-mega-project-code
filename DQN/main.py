
import gymnasium as gym
import ale_py #needed for namespace
import cv2
from gymnasium.wrappers import FrameStackObservation
import numpy as np
from DQN import phi
from DQN import DQN
import torch
import random


def populate_buffer(env,replay_buffer,frame_skip,amount_to_pop):
    """
    Takes in env, buffer, frames to skip, and amount to populate (in number of frames). 

    returns: total_frame_count

    populates replay buffer with some transition dynamics
    """
    phi_1 = 0
    phi_2 = 0

    #phi_1 = torch.tensor(phi(frames),dtype=torch.float)

    total_frame_count = 0
    #populate replay buffer with random plays, split into helper function later
    frames = []
    observation, _ = env.reset()
    frames.append(observation)
    total_frame_count += 1
    old_action = 0
    old_reward = 0

    action = 0 #start with no action

    for i in range(3): #setting up
        total_frame_count += 1
        observation, reward, terminated, truncated, _ = env.step(action)
        frames.append(observation)
        if len(frames) == frame_skip:
            action = env.action_space.sample() 
            phi_1 = phi(frames)
            phi_2 = phi_1

            old_action = action
            old_reward = reward
            frames = []

    while total_frame_count <= amount_to_pop:
        observation, reward, terminated, truncated, _ = env.step(action)
        frames.append(observation)
        total_frame_count += 1

        if terminated or truncated:
            phi_1 = phi_2
            phi_2 = -1
            transition = (phi_1,old_action,old_reward,phi_2)
            replay_buffer.append(transition)

            frames = []
            observation, _ = env.reset()
            frames.append(observation)
            action = env.action_space.sample() 
            old_action = action
            old_reward = 0


        if len(frames) == frame_skip:
            # Check if this is the first state after a reset (phi_2 is -1)
            if not isinstance(phi_2, np.ndarray) and phi_2 == -1:
                phi_1 = phi(frames)
                phi_2 = phi_1
                frames = []
                continue
            else:
                phi_1 = phi_2
                phi_2 = phi(frames) 
            transition = (phi_1,old_action,old_reward,phi_2)
            replay_buffer.append(transition)

            action =  env.action_space.sample() 
            old_action = action
            old_reward = reward
            frames = []
    

    return  total_frame_count 

def eps_anneal(initial, final, total_frames):
    """
    Returns a function that calculates epsilon for a given frame.
    """
    step_size = (initial - final) / total_frames
    
    def get_epsilon(current_frame):
        epsilon = initial - step_size * current_frame
        return max(final, min(initial, epsilon))
    
    return get_epsilon

def breakout_training():
    #render_mode="human" for when I want to watch an episode
    env = gym.make('ALE/Breakout-v5')

    replay_buffer = []
    total_frame_count = 0 

    behavior_model = DQN(4) #4 is output actions
    target_model = DQN(4)

    lr = .05
    optimizer = torch.optim.Adam(behavior_model.parameters(),lr=lr)
    MSE_loss = torch.nn.MSELoss()
    #target network updare frequence.
    network_update_freq = 1000

    #Do SGD updates after this many actions
    update_frequency = 4 
    action_count = 0

    #other hyper params
    episodes = 1
    discount = .99
    total_frame_count = 0 
    batch_size = 2

    #eps annealing hardcode, for 100k frames rather than 1M like in paper
    eps_val = eps_anneal(1,.1,100)


    episode_over = False
    
    episode_rewards = []

    frame_skip = 4

    phi_1 = 0
    phi_2 = 0

    total_frame_count = populate_buffer(env,replay_buffer,frame_skip,20) 


    for i in range(episodes): 
        episode_over = False

        frames = []
        observation, _ = env.reset()
        frames.append(observation)
        total_frame_count += 1

        phi_2 = -1 #make sure 
        action = 0
        old_action = 0
        old_reward = 0
        total_reward = 0

        while not episode_over:
            observation, reward, terminated, truncated, _ = env.step(action)
            total_frame_count += 1
            frames.append(observation)

            if len(frames)== frame_skip:
                if not isinstance(phi_2, np.ndarray) and phi_2 == -1:
                    phi_1 = phi(frames)
                    phi_2 = phi_1
                    frames = []
                    continue
                else:
                    phi_1 = phi_2
                    phi_2 = phi(frames) 
                transition = (phi_1,old_action,old_reward,phi_2)
                replay_buffer.append(transition)


                #select e-greedy action
                if np.random.random() < eps_val(total_frame_count):
                    action = np.random.randint(0, 3)
                else:

                    processed_frames = torch.tensor(phi_2,dtype=torch.float)
                    action = torch.argmax(behavior_model(processed_frames)).item()


                #check if it's SGD time
                if action_count == update_frequency:
                    optimizer.zero_grad()
                    action_count = 0


                    minibatch = random.choices(replay_buffer,k = batch_size)

                    target_Qs = []
                    pred_Q = []

                    for i in range(len(minibatch)):
                        if not isinstance(minibatch[i][-1], np.ndarray) and minibatch[i][-1] == -1:
                            target_Qs.append(minibatch[i][2])
                        else:
                            with torch.no_grad():
                                target_frames = torch.tensor(minibatch[i][-1],dtype=torch.float)
                                target_model_val = torch.max(target_model(target_frames)).item()
                                reward = minibatch[i][2]
                                target_Qs.append(reward + discount * target_model_val)

                        pred_frames = torch.tensor(minibatch[i][0],dtype=torch.float)
                        action_taken = minibatch[i][1]
                        q_values = behavior_model(pred_frames)
                        pred_Q.append(q_values[0, action_taken]) #nothing says we take the max here in the paper pseudocode.

                    ys = torch.tensor(target_Qs,dtype=torch.float)
                    pds = torch.stack(pred_Q)
                    loss = MSE_loss(pds,ys)

                    loss.backward()
                    optimizer.step()


                old_action = action
                old_reward = reward
                frames = []
                action_count += 1

            
            #update target model every C frames
            if total_frame_count % network_update_freq == 0:
                print("copying over model")
                target_model.load_state_dict(behavior_model.state_dict())

            total_reward += reward
            episode_over = terminated or truncated

        episode_rewards.append(total_reward)


    env.close()
    return episode_rewards


reward_list = breakout_training()
print(reward_list)