import sys
print("Python path:")
for path in sys.path:
    print(f"  {path}")

print("\nTrying to import our modules:")
try:
    from src.agents.dqn_agent import DQN
    print("Successfully imported DQN")
except Exception as e:
    print(f"Failed to import DQN: {e}")

try:
    from src.utils.preprocessing import preprocess_frame
    print("Successfully imported preprocess_frame")
except Exception as e:
    print(f"Failed to import preprocess_frame: {e}")

try:
    from src.config.hyperparameters import BATCH_SIZE
    print("Successfully imported hyperparameters")
except Exception as e:
    print(f"Failed to import hyperparameters: {e}") 