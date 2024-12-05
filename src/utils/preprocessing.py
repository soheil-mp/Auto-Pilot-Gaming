import cv2
import numpy as np
import torch

def preprocess_frame(frame):
    """Convert frame to grayscale and resize it."""
    if len(frame.shape) == 1:
        # Handle 1D state space (like CartPole)
        return frame
    elif len(frame.shape) == 2:
        # Already grayscale
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        return frame / 255.0
    else:
        # RGB to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        return frame / 255.0

def stack_frames(frames):
    """Stack multiple frames together."""
    if isinstance(frames[0], (int, float)) or len(frames[0].shape) == 1:
        # Handle 1D state space
        return torch.FloatTensor(frames)
    return torch.FloatTensor(np.stack(frames, axis=0)) 