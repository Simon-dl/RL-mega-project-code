import matplotlib.pyplot as plt
import numpy as np

# Scratchpad

reward_list = [1,2,4,5,6,7,8]
episodes = np.arange(0, len(reward_list), 1)

plt.plot(episodes, reward_list)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Rewards vs Episodes')
plt.grid(True)
plt.show()
