import torch
import numpy as np
import cv2




#Preprocessing function
def phi(frames):
    """
    Preprocess the input frames to a stack of resized grayscale images.

    expect frames to be a tensor of shape (n_frames, height, width, channels)
    return a tensor of shape (84, 84, n_frames )
    """
    return np.stack([cv2.resize(frame, (84, 84)) for frame in frames], axis=0).squeeze(1)


test_frames = np.random.rand(4, 210, 160, 3)
print(phi(test_frames).shape)

#Model
# class DQN(torch.nn.Module):
#     def __init__(self, input_shape, n_actions):
#         super(DQN, self).__init__()
#         self.conv = torch.nn.Sequential(
#             torch.nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
#             torch.nn.ReLU(),
#             torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
#             torch.nn.ReLU(),
#             torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
#             torch.nn.ReLU()
#         )
#         self.fc = torch.nn.Sequential(