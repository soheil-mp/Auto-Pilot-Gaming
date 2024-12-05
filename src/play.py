import argparse
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import os
import sys
from datetime import datetime

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.agents.dqn_agent import DQN
from src.utils.preprocessing import preprocess_frame
from src.config.hyperparameters import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', required=True, help='Environment name')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes')
    parser.add_argument('--render', type=bool, default=True, help='Enable visualization')
    parser.add_argument('--record', action='store_true', help='Record video of the agent playing')
    parser.add_argument('--video-dir', default='videos', help='Directory to save videos')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create environment
    env = gym.make(args.env, render_mode='rgb_array' if args.record else ('human' if args.render else None))
    
    # Wrap environment with video recorder if recording is enabled
    if args.record:
        videos_dir = os.path.join(project_root, args.video_dir)
        env = RecordVideo(
            env,
            videos_dir,
            episode_trigger=lambda x: True,  # Record all episodes
            name_prefix=f"{args.env}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        print(f"Recording videos to: {videos_dir}")
    
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
    
    # Load model
    model = DQN(input_shape, n_actions).to(device)
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded model from episode {checkpoint['episode']} with average reward {checkpoint['avg_reward']:.2f}")
    
    total_rewards = []
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
                # Select action
                with torch.no_grad():
                    action = model(state).max(1)[1].item()
                
                # Take action
                next_state, reward, done, truncated, _ = env.step(action)
                total_reward += reward
                
                next_state = preprocess_frame(next_state)
                if isinstance(next_state, (int, float)) or len(next_state.shape) == 1:
                    next_state = torch.FloatTensor(next_state).unsqueeze(0)
                else:
                    next_state = torch.cat([state[0, 1:], torch.FloatTensor([next_state])], 0)
                    next_state = next_state.unsqueeze(0)
                
                state = next_state
            
            total_rewards.append(total_reward)
            print(f"Episode {episode + 1}/{args.episodes}, Total Reward: {total_reward}")
    
    except KeyboardInterrupt:
        print("\nVisualization interrupted by user")
    
    finally:
        env.close()
        avg_reward = sum(total_rewards) / len(total_rewards)
        print(f"\nAverage Reward over {len(total_rewards)} episodes: {avg_reward:.2f}")
        if args.record:
            print(f"Videos have been saved to: {videos_dir}")

if __name__ == '__main__':
    main() 