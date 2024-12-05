import matplotlib.pyplot as plt
import numpy as np
from collections import deque

class TrainingVisualizer:
    def __init__(self, window_size=100):
        self.rewards = []
        self.avg_rewards = []
        self.window_size = window_size
        self.reward_window = deque(maxlen=window_size)
        
        # Setup plot
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.line1, = self.ax.plot([], [], 'b-', label='Reward')
        self.line2, = self.ax.plot([], [], 'r-', label='Average Reward')
        self.ax.set_xlabel('Episode')
        self.ax.set_ylabel('Reward')
        self.ax.set_title('Training Progress')
        self.ax.legend()
        plt.show(block=False)
    
    def update(self, episode, reward):
        self.rewards.append(reward)
        self.reward_window.append(reward)
        avg_reward = np.mean(self.reward_window)
        self.avg_rewards.append(avg_reward)
        
        # Update plot
        self.line1.set_data(range(len(self.rewards)), self.rewards)
        self.line2.set_data(range(len(self.avg_rewards)), self.avg_rewards)
        
        # Adjust plot limits
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        return avg_reward
    
    def close(self):
        plt.close() 