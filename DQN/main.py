
import gymnasium as gym
import ale_py #needed for namespace
import cv2
from gymnasium.wrappers import FrameStackObservation
import numpy as np
from DQN import phi
from DQN import DQN
import torch



def populate_buffer(env,replay_buffer,frame_skip,amount_to_pop):
    """
    Takes in env, buffer, frames to skip, and amount to populate (in number of frames). 

    returns: phi_2 and total_frame_count
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
                        # Check if this is the first state after a reset (phi_2 is -1 or None)
            if phi_2 == -1:
                # First state of new episode: both phi_1 and phi_2 are the same
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
    

    return phi_2, total_frame_count 

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

    behavior_model = DQN(4)
    target_model = DQN(4)

    lr = .05
    optimizer = torch.optim.Adam(behavior_model.parameters(),lr=lr)

    #target network updare frequence.
    network_update_freq = 1000
    C = 0

    #Do SGD updates after this many actions
    update_frequency = 4 
    action_count = 0

    #other hyper params
    episodes = 1
    discount = .99
    total_frame_count = 0 
    batch_size = 2

    #eps annealing hardcode, for 100k frames rather than 1M like in paper
    eps_val = eps_anneal(1,.1,1000)


    episode_over = False
    
    episode_rewards = []

    frame_skip = 4

    phi_1 = 0
    phi_2 = 0

    phi_2, total_frame_count = populate_buffer(env,replay_buffer,frame_skip,20) 

    print(" \n phi_2",phi_2)
    print(" \n total_frame_count", total_frame_count)
    print(" \n buffer amount",len(replay_buffer))
    print(" \n buffer 1 action",replay_buffer[1][1])


    for i in range(episodes): 
        episode_over = False

        frames = []
        observation, _ = env.reset()
        frames.append(observation)
        total_frame_count += 1

        old_action = 0
        old_reward = 0
        total_reward = 0

        while not episode_over:
            observation, reward, terminated, truncated, _ = env.step(action)
            total_frame_count += 1
            frames.append(observation)

            if len(frames)== frame_skip:
                if phi_2 == -1:
                    # First state of new episode: both phi_1 and phi_2 are the same
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
                

                processed_frames = torch.tensor(phi_2,dtype=torch.float)
                action = torch.argmax(behavior_model(processed_frames)).item()

            total_reward += reward
            episode_over = terminated or truncated

        episode_rewards.append(total_reward)
    env.close()

breakout_training()