
import gymnasium as gym
import ale_py #needed for namespace
import cv2
import numpy as np
from DQN import phi
from DQN import DQN
import torch


def breakout_training():
    #render_mode="human" for when I want to watch an episode
    env = gym.make('ALE/Breakout-v5')

    replay_memory = []

    behavior_model = DQN(4)
    target_model = DQN(4)

    lr = .05
    optimizer = torch.optim.Adam(behavior_model.parameters(),lr=lr)

    update_target = 20

    episodes = 1
    discount = .99


    for i in range(episodes): 
        episode_over = False
        total_reward = 0

        frame_skip = 4


        phi_1 = 0
        phi_2 = 0



        
        #get inital proccessed frames
        frames = []
        action = 0 #start with no action
        observations = env.reset()
        frames.append(observations)
        for j in range(3):
            observation, reward, terminated, truncated = env.step(action)
            frames.append(observation)
        phi_1 = torch.tensor(phi(frames),dtype=torch.float)
        action = torch.argmax(behavior_model(phi_1)).item() #should ideally do eps-greedy action selection here
        


        frame_counter = 0 
        # while not episode_over:
        for i in range(4):
            print(i)
            observation, reward, terminated, truncated = env.step(action)

            frames.append(observation)
            frame_counter += 1 
            if frame_counter == frame_skip:
                print("here", frame_counter)
                frame_counter = 0
                processed_frames = torch.tensor(phi(frames),dtype=torch.float)
                print(processed_frames.shape)
                action = torch.argmax(behavior_model(processed_frames)).item()
                print("action",action)





            total_reward += reward
            episode_over = terminated or truncated

        print(f"Episode finished! Total reward: {total_reward}")
    env.close()

breakout_training()