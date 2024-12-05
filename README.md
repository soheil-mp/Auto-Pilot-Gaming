<div align="center">

# 🎮 Auto Pilot Gaming

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-Latest-orange?style=for-the-badge)](https://gymnasium.farama.org/)

<p align="center">
    <em>A deep reinforcement learning framework for training AI agents in classic control and Atari environments. 🚀</em>
</p>

[Features](#features) • [Installation](#-installation) • [Usage](#-usage) • [Documentation](#-documentation) • [Contributing](#-contributing)

---

</div>

## 📁 Project Structure

```
adaptive-game-ai/
├── src/
│   ├── agents/
│   │   └── dqn_agent.py     # DQN implementation
│   ├── utils/
│   │   ├── preprocessing.py  # Frame processing utilities
│   │   ├── replay_buffer.py  # Experience replay implementation
│   │   └── visualization.py  # Training visualization
│   ├── config/
│   │   └── hyperparameters.py # Training configurations
│   └── play.py              # Script to run trained agents
├── models/                  # Saved model checkpoints
├── logs/                    # Training logs
├── videos/                  # Recorded gameplay videos
├── tests/                   # Unit tests
└── requirements.txt         # Project dependencies
```

## ✨ Features

### 🧠 Core Features
- Deep Q-Learning (DQN) implementation
- Experience replay mechanism
- Frame stacking & preprocessing
- ε-greedy exploration strategy
- Real-time training visualization
- Video recording of trained agents

### 🎯 Capabilities
- Multi-game support (both classic control and Atari games)
- Model checkpointing with best model saving
- Real-time visualization of training progress
- Performance metrics tracking
- Gameplay video recording
- Support for both image-based and vector-based environments

## 🛠️ Tech Stack

### Core Dependencies
- Python 3.8+
- PyTorch 2.0+
- Gymnasium[atari] 0.29+
- NumPy 1.24+
- OpenCV 4.8+
- Matplotlib 3.7+

### Optional Tools
- CUDA-capable GPU
- MoviePy (for video recording)
- Pygame (for environment rendering)

## 📋 Prerequisites

Before you begin, ensure you have the following:
- ✅ Python 3.8 or higher installed
- ✅ pip (Python package manager)
- ✅ Virtual environment (recommended)
- ✅ CUDA-capable GPU (optional, for faster training)

## 🚀 Installation

1. **Clone the repository:**

```bash
git clone [repository-url]
cd adaptive-game-ai
```

2. **Set up virtual environment:**

```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# Unix/MacOS
source venv/bin/activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt

# For video recording support
pip install "gymnasium[other]" moviepy
```

## 🎮 Usage

### Training Your AI

```bash
python -m src.agents.dqn_agent --env CartPole-v1 \
                              --episodes 1000 \
                              --save-path models/cartpole_dqn.pt \
                              --target-reward 195
```

Available Training Options:

| Option | Description | Default |
|--------|-------------|---------|
| `--env` | Environment name | Required |
| `--episodes` | Number of training episodes | 1000 |
| `--save-path` | Model save location | Required |
| `--device` | Training device (cuda/cpu) | auto |
| `--target-reward` | Target reward to consider solved | 195.0 |

### Playing Games

```bash
python -m src.play --env CartPole-v1 \
                   --model models/cartpole_dqn.pt \
                   --episodes 5 \
                   --record \
                   --video-dir videos/cartpole
```

Available Play Options:

| Option | Description | Default |
|--------|-------------|---------|
| `--env` | Environment name | Required |
| `--model` | Path to trained model | Required |
| `--episodes` | Number of episodes | 5 |
| `--render` | Enable visualization | True |
| `--record` | Record gameplay videos | False |
| `--video-dir` | Directory to save videos | videos |

### Video Recording

The agent's gameplay can be recorded using the `--record` flag. Videos are saved in MP4 format with the following naming convention:
```
{env_name}_{timestamp}-episode-{episode_number}.mp4
```

Example video directory structure:
```
videos/
└── cartpole_test/
    ├── CartPole-v1_20240101_120000-episode-0.mp4
    ├── CartPole-v1_20240101_120000-episode-1.mp4
    └── CartPole-v1_20240101_120000-episode-2.mp4
```

## 🧪 Implementation Details

Our implementation leverages state-of-the-art techniques in deep reinforcement learning:

- 🔄 DQN with experience replay buffer
- 🖼️ CNN architecture for image processing
- 📊 Advanced reward shaping
- 🎯 Frame stacking (4 frames)
- 🔍 Epsilon-greedy exploration
- 📈 Real-time training visualization
- 🎥 Gameplay video recording

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
