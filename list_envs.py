import gymnasium as gym
from gymnasium.envs.registration import registry

print("Available environments:")
for env_spec in registry.values():
    print(f"- {env_spec.id}") 