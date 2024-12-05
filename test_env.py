import gymnasium as gym

env = gym.make('CartPole-v1')
print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")

state = env.reset()[0]
print(f"Initial state: {state}")

for _ in range(10):
    action = env.action_space.sample()
    state, reward, done, truncated, info = env.step(action)
    print(f"Action: {action}, Reward: {reward}, Done: {done}")
    if done:
        break

env.close() 