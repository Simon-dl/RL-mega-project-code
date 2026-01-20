# Run `pip install "gymnasium[classic-control]"` for this example.
import gymnasium as gym
# Create our training environment - a cart with a pole that needs balancing
env = gym.make("'Walker2d-v5'", render_mode="human")

# Reset environment to start a new episode
observation, info = env.reset()

print(f"Starting observation: {observation}")

episode_over = False
total_reward = 0

while not episode_over:
    action = env.action_space.sample()  

    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    episode_over = terminated or truncated

print(f"Episode finished! Total reward: {total_reward}")
env.close()