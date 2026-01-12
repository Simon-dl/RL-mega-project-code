
import gymnasium as gym
import ale_py #needed for namespace

env = gym.make('ALE/Breakout-v5',render_mode="human")

episode_over = False
total_reward = 0

print(env.action_space)

observations, info = env.reset()

while not episode_over:
    action = env.action_space.sample()  # Random action for now - real agents will be smarter!

    # Take the action and see what happens
    observation, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    episode_over = terminated or truncated

print(f"Episode finished! Total reward: {total_reward}")
env.close()