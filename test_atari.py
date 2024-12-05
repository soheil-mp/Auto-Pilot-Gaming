import gymnasium as gym

env = gym.make("PongNoFrameskip-v4", render_mode="human")
print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")

state = env.reset()[0]
print(f"Initial state shape: {state.shape}")

for _ in range(100):
    action = env.action_space.sample()
    state, reward, done, truncated, info = env.step(action)
    if done:
        break

env.close() 