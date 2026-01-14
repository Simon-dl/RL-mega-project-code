
import gymnasium as gym
import ale_py #needed for namespace
import cv2
import numpy as np


#render_mode="human" for when I want to watch an episode
env = gym.make('ALE/Breakout-v5')

episode_over = False
total_reward = 0
frames = []

print(env.action_space)

observations, info = env.reset()
frames.append(observations)
print(observations.shape)

# while not episode_over:
for i in range(3):
    action = env.action_space.sample() 
    
    observation, reward, terminated, truncated, info = env.step(action)
    frames.append(observation)

    total_reward += reward
    episode_over = terminated or truncated

print(f"Episode finished! Total reward: {total_reward}")
env.close()

def phi(frames):
    """
    Preprocess the input frames to a stack of resized grayscale images.

    expect frames to be a tensor of shape (n_frames, height, width, channels)
    return a tensor of shape (84, 84, n_frames )
    """
    new_frames = []
    for i in range(len(frames)):
        frame = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        new_frames.append(cv2.resize(frame, (84, 84)))
    return np.array(new_frames)


frames = np.array(frames)
print(frames.shape)

cv2.imshow('frame', observations)
cv2.waitKey(0)
cv2.destroyAllWindows()

out = phi(frames)
print("out shape: ", out.shape)
cv2.imshow('frame', out[0])
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('frame', out[1])
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('frame', out[2])
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('frame', out[3])
cv2.waitKey(0)
cv2.destroyAllWindows()