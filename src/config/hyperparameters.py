"""Hyperparameters for DQN training."""

# Training parameters
BATCH_SIZE = 32
GAMMA = 0.99
LEARNING_RATE = 1e-4
EPSILON_START = 1.0
EPSILON_FINAL = 0.01
EPSILON_DECAY = 50000
TARGET_UPDATE = 1000
MEMORY_SIZE = 10000
NUM_FRAMES = 4

# Environment parameters
FRAME_SIZE = (84, 84)
FRAME_SKIP = 4

# Model parameters
HIDDEN_SIZE = 512
KERNEL_SIZES = [8, 4, 3]
STRIDES = [4, 2, 1]
CHANNELS = [32, 64, 64] 