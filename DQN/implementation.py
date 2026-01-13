
import gymnasium as gym
import ale_py #needed for namespace
import torch


#render_mode="human" for when I want to watch an episode
env = gym.make('ALE/Breakout-v5')

episode_over = False
total_reward = 0

print(env.action_space)

observations, info = env.reset()

print(observations.shape)

while not episode_over:
    action = env.action_space.sample() 
    
    observation, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    episode_over = terminated or truncated

print(f"Episode finished! Total reward: {total_reward}")
env.close()