from collections import deque
import random
import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        """Initialize replay buffer.
        
        Args:
            capacity (int): Maximum size of the buffer
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Store transition in the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a batch of transitions.
        
        Args:
            batch_size (int): Size of batch to sample
            
        Returns:
            tuple: Batch of transitions (state, action, reward, next_state, done)
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        
        return (torch.cat(state),
                torch.tensor(action),
                torch.tensor(reward, dtype=torch.float32),
                torch.cat(next_state),
                torch.tensor(done, dtype=torch.float32))
    
    def __len__(self):
        """Return current size of the buffer."""
        return len(self.buffer) 