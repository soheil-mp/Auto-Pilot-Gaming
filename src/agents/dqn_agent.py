import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from pathlib import Path
import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.utils.preprocessing import preprocess_frame, stack_frames
from src.utils.replay_buffer import ReplayBuffer
from src.utils.visualization import TrainingVisualizer
from src.config.hyperparameters import *

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        
        if isinstance(input_shape, int):
            # For environments with 1D state space
            self.net = nn.Sequential(
                nn.Linear(input_shape, HIDDEN_SIZE),
                nn.ReLU(),
                nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
                nn.ReLU(),
                nn.Linear(HIDDEN_SIZE, n_actions)
            )
        else:
            # For environments with image observations
            self.conv = nn.Sequential(
                nn.Conv2d(input_shape[0], CHANNELS[0], kernel_size=KERNEL_SIZES[0], stride=STRIDES[0]),
                nn.ReLU(),
                nn.Conv2d(CHANNELS[0], CHANNELS[1], kernel_size=KERNEL_SIZES[1], stride=STRIDES[1]),
                nn.ReLU(),
                nn.Conv2d(CHANNELS[1], CHANNELS[2], kernel_size=KERNEL_SIZES[2], stride=STRIDES[2]),
                nn.ReLU()
            )
            
            conv_out_size = self._get_conv_out(input_shape)
            self.fc = nn.Sequential(
                nn.Linear(conv_out_size, HIDDEN_SIZE),
                nn.ReLU(),
                nn.Linear(HIDDEN_SIZE, n_actions)
            )
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        if hasattr(self, 'conv'):
            conv_out = self.conv(x).view(x.size()[0], -1)
            return self.fc(conv_out)
        else:
            return self.net(x)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', required=True, help='Atari environment name')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--save-path', required=True, help='Path to save model')
    parser.add_argument('--device', default='auto', help='Device to use (cuda/cpu)')
    parser.add_argument('--target-reward', type=float, default=195.0, help='Target average reward to consider solved')
    args = parser.parse_args()
    
    # Create save directory if it doesn't exist
    save_dir = Path(args.save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')
    print(f"Using device: {device}")
    
    # Create environment
    env = gym.make(args.env)
    n_actions = env.action_space.n
    print(f"Environment: {args.env}, Action space: {n_actions}")
    
    # Determine input shape
    state = env.reset()[0]
    if isinstance(state, (int, float)) or len(state.shape) == 1:
        input_shape = state.shape[0] if hasattr(state, 'shape') else 1
        print(f"Using MLP for 1D state space of size {input_shape}")
    else:
        input_shape = (NUM_FRAMES, *FRAME_SIZE)
        print(f"Using CNN for image input of shape {input_shape}")
    
    # Initialize DQN and target network
    policy_net = DQN(input_shape, n_actions).to(device)
    target_net = DQN(input_shape, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    
    # Setup training
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayBuffer(MEMORY_SIZE)
    frame_idx = 0
    visualizer = TrainingVisualizer(window_size=100)
    best_avg_reward = float('-inf')
    
    try:
        for episode in range(args.episodes):
            state = env.reset()[0]
            state = preprocess_frame(state)
            if isinstance(state, (int, float)) or len(state.shape) == 1:
                state = torch.FloatTensor(state).unsqueeze(0).to(device)
            else:
                state = torch.FloatTensor([state for _ in range(NUM_FRAMES)]).unsqueeze(0).to(device)
            
            total_reward = 0
            done = False
            
            while not done:
                epsilon = EPSILON_FINAL + (EPSILON_START - EPSILON_FINAL) * \
                         np.exp(-1. * frame_idx / EPSILON_DECAY)
                frame_idx += 1
                
                # Select action
                if np.random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = policy_net(state).max(1)[1].item()
                
                # Take action
                next_state, reward, done, truncated, _ = env.step(action)
                total_reward += reward
                
                next_state = preprocess_frame(next_state)
                if isinstance(next_state, (int, float)) or len(next_state.shape) == 1:
                    next_state = torch.FloatTensor(next_state).unsqueeze(0)
                else:
                    next_state = torch.cat([state[0, 1:], torch.FloatTensor([next_state])], 0)
                    next_state = next_state.unsqueeze(0)
                
                # Store transition
                memory.push(state, action, reward, next_state, done)
                state = next_state
                
                # Train if enough samples
                if len(memory) >= BATCH_SIZE:
                    states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)
                    states = states.to(device)
                    actions = actions.to(device)
                    rewards = rewards.to(device)
                    next_states = next_states.to(device)
                    dones = dones.to(device)
                    
                    # Compute Q values
                    current_q = policy_net(states).gather(1, actions.unsqueeze(1))
                    next_q = target_net(next_states).max(1)[0].detach()
                    target_q = rewards + (1 - dones) * GAMMA * next_q
                    
                    # Compute loss and update
                    loss = nn.MSELoss()(current_q.squeeze(), target_q)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                if frame_idx % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())
            
            # Update visualization
            avg_reward = visualizer.update(episode, total_reward)
            print(f"Episode {episode + 1}/{args.episodes}, Total Reward: {total_reward}, Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon:.3f}")
            
            # Save best model
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                torch.save({
                    'episode': episode,
                    'model_state_dict': policy_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'avg_reward': avg_reward,
                }, args.save_path)
                print(f"New best model saved with average reward: {avg_reward:.2f}")
            
            # Check if environment is solved
            if avg_reward >= args.target_reward:
                print(f"\nEnvironment solved in {episode + 1} episodes!")
                print(f"Average reward: {avg_reward:.2f}")
                break
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    finally:
        env.close()
        visualizer.close()
        print(f"\nBest average reward: {best_avg_reward:.2f}")

if __name__ == '__main__':
    main() 