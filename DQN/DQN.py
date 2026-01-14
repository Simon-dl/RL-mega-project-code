import torch
import numpy as np
import cv2



#preprocessing function for stack of frames
def phi(frames):
    """
    Preprocess the input frames to a stack of resized grayscale images.

    expect frames to be a tensor of shape (n_frames, height, width, channels)
    return a tensor of shape (84, 84, n_frames )
    """
    frames = np.array(frames)
    new_frames = []
    for i in range(len(frames)):
        frame = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        new_frames.append(cv2.resize(frame, (84, 84)))
    return np.array(new_frames).reshape(1,4,84,84)

#Model
class DQN(torch.nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.ReLU()
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_shape[0],512),
            torch.nn.Linear(512,n_actions)
        )
    
    def forward(self, x):
        out_conv = self.conv(x)
        return self.fc(out_conv)