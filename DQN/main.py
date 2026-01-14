
import gymnasium as gym
import ale_py #needed for namespace
import cv2
import numpy as np
from DQN import phi
import torch


class DQN(torch.nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(input_shape[1], 32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.ReLU()
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(3136,512), #hardcoded the elements gotten by flattened with out_conv.numel()
            torch.nn.Linear(512,n_actions)
        )
    
    def forward(self, x):
        out_conv = self.conv(x)
        print(out_conv.numel())
        x = torch.flatten(out_conv, start_dim=1)
        return self.fc(x)


#render_mode="human" for when I want to watch an episode
env = gym.make('ALE/Breakout-v5')

model = DQN((1,4,84,84),4)
episode_over = False
total_reward = 0

frame_skip = 4
frames = []
frame_counter = 1

print(env.action_space)

observations, info = env.reset()
frames.append(observations)
print(observations.shape)

action = env.action_space.sample() 

# while not episode_over:
for i in range(4):
    print(i)
    observation, reward, terminated, truncated, info = env.step(action)

    frames.append(observation)
    frame_counter += 1 
    if frame_counter == frame_skip:
        print("here", frame_counter)
        frame_counter = 0
        processed_frames = torch.tensor(phi(frames),dtype=torch.float)
        print(processed_frames.shape)
        actions = model(processed_frames)
        print(actions)





    total_reward += reward
    episode_over = terminated or truncated

print(f"Episode finished! Total reward: {total_reward}")
env.close()

