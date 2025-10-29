# Deep Reinforcement Learning Algorithms

This project implements multiple state-of-the-art Deep Reinforcement Learning (DRL) algorithms to solve the CartPole-v1 environment from OpenAI Gym. The implementations include REINFORCE, DQN, PPO, and DPO, providing a comprehensive comparison of different RL approaches.

## 📚 Algorithms Implemented

### 1. REINFORCE (1992)
- **Type**: Policy Gradient
- **File**: `reinforce_cartpole.py`
- **Features**: Monte Carlo policy gradient, baseline normalization
- **Best for**: Learning RL fundamentals

### 2. DQN (2013)
- **Type**: Value-based Learning
- **File**: `dqn_cartpole.py`
- **Features**: Experience replay, target network, ε-greedy exploration
- **Best for**: Discrete action spaces, high sample efficiency

### 3. PPO (2017)
- **Type**: Policy Gradient with Trust Region
- **File**: `ppo_cartpole.py`
- **Features**: Clipped probability ratio, GAE, Actor-Critic
- **Best for**: Production environments, robust training

### 4. DPO (2023)
- **Type**: Direct Preference Optimization
- **File**: `dpo_cartpole.py`
- **Features**: Preference learning, reference policy, no reward modeling
- **Best for**: Learning from human preferences, LLM alignment

## 📂 Project Structure

```
Deep-Reinforcement-Learning/
├── reinforce_cartpole.py      # REINFORCE implementation
├── dqn_cartpole.py             # DQN implementation
├── ppo_cartpole.py             # PPO implementation
├── dpo_cartpole.py             # DPO implementation
├── docs/                       # Algorithm documentation
│   ├── REINFORCE_EXPLANATION.md
│   ├── DQN_EXPLANATION.md
│   ├── PPO_EXPLANATION.md
│   ├── DPO_EXPLANATION.md
│   └── ALGORITHMS_COMPARISON.md
├── output/                     # Training outputs
│   ├── reinforce/
│   ├── dqn/
│   ├── ppo/
│   └── dpo/
└── README.md
```

## Setup

To set up the project, follow these steps:

1. **Clone the repository (if applicable):**
   ```bash
   git clone <repository_url>
   cd Deep-Reinforcement-Learning
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows
   # source venv/bin/activate    # On macOS/Linux
   ```

3. **Install dependencies:**
   ```bash
   pip install torch gymnasium matplotlib numpy pandas
   ```

## Usage

### Running Individual Algorithms

Each algorithm can be run independently:

```bash
# REINFORCE - Simple policy gradient
python reinforce_cartpole.py

# DQN - Value-based learning
python dqn_cartpole.py

# PPO - Advanced policy gradient (recommended)
python ppo_cartpole.py

# DPO - Preference-based learning
python dpo_cartpole.py
```

Each script will:
- Initialize the CartPole environment and the corresponding agent
- Train the agent, saving checkpoints periodically
- Save training history as CSV and generate plots
- Evaluate the trained agent with real-time visualization

### Quick Start

For first-time learners, recommended order:
1. **REINFORCE** - Understand basic policy gradient
2. **DQN** - Learn value-based methods
3. **PPO** - Master state-of-the-art algorithm
4. **DPO** - Explore preference learning

## Visualization

The `evaluate_and_visualize` function displays a 2x2 grid with the following subplots:

- **Top-Left:** Real-time rendering of the CartPole environment.
- **Top-Right:** Cumulative reward curve over evaluation steps.
- **Bottom (spanning both columns):** A square wave representation of the agent's actions (Left/Right) over the last 500 steps.

You can adjust the `render_interval` parameter in the `evaluate_and_visualize` function to control the visualization speed.

## Checkpoints and Output

Each algorithm saves its outputs in a separate directory:

```
output/
├── reinforce/
│   ├── reinforce_checkpoint.pth
│   ├── reinforce_cartpole_training_history.csv
│   └── reinforce_cartpole_training_history.png
├── dqn/
│   ├── dqn_checkpoint.pth
│   ├── dqn_cartpole_training_history.csv
│   └── dqn_cartpole_training_history.png
├── ppo/
│   ├── ppo_checkpoint.pth
│   ├── ppo_cartpole_training_history.csv
│   └── ppo_cartpole_training_history.png
└── dpo/
    ├── dpo_checkpoint.pth
    ├── dpo_cartpole_training_history.csv
    └── dpo_cartpole_training_history.png
```

## 📖 Documentation

Detailed explanations for each algorithm are available in the `docs/` folder:

- **[REINFORCE_EXPLANATION.md](docs/REINFORCE_EXPLANATION.md)** - Comprehensive guide to the REINFORCE algorithm
- **[DQN_EXPLANATION.md](docs/DQN_EXPLANATION.md)** - Deep dive into Deep Q-Networks
- **[PPO_EXPLANATION.md](docs/PPO_EXPLANATION.md)** - Complete PPO tutorial
- **[DPO_EXPLANATION.md](docs/DPO_EXPLANATION.md)** - Understanding Direct Preference Optimization
- **[ALGORITHMS_COMPARISON.md](docs/ALGORITHMS_COMPARISON.md)** - Side-by-side comparison of all algorithms

Each documentation includes:
- ✅ Core concepts and theory
- ✅ Algorithm pseudocode
- ✅ Implementation details
- ✅ Hyperparameter tuning guide
- ✅ Training tips and debugging
- ✅ Use cases and applications

## 🎯 Performance Comparison (CartPole-v1)

Based on 1000 training episodes:

| Algorithm | Avg Reward | Convergence Speed | Stability | Sample Efficiency |
|-----------|-----------|-------------------|-----------|------------------|
| REINFORCE | ~380 | ⭐⭐ | ⭐⭐ | ⭐⭐ |
| DQN | ~410 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| PPO | ~475 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| DPO | ~440 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

**Note**: CartPole-v1 max reward is 500. Results may vary based on hyperparameters and random seeds.

## 🔑 Key Features

### All Implementations Include:
- ✅ Checkpoint saving and loading
- ✅ Training history tracking
- ✅ Automatic visualization generation
- ✅ Real-time evaluation with plots
- ✅ Comprehensive logging
- ✅ Compatible with Gymnasium

### Algorithm-Specific Features:

**REINFORCE**
- Baseline normalization
- Monte Carlo returns
- Simple and educational

**DQN**
- Experience replay buffer
- Target network
- ε-greedy exploration with decay

**PPO**
- Clipped probability ratio
- Generalized Advantage Estimation (GAE)
- Actor-Critic architecture
- Multi-epoch updates

**DPO**
- Preference pair learning
- Reference policy constraint
- Direct optimization (no reward model)
- Trajectory comparison

## 🛠️ Troubleshooting

### Common Issues

**Import Errors**
```bash
# Make sure gymnasium is installed (not gym)
pip install gymnasium torch matplotlib pandas numpy
```

**CUDA Issues**
```python
# The code automatically detects and uses GPU if available
# To force CPU, modify the device line in each script:
self.device = torch.device("cpu")
```

**Training Not Converging**
- Check the algorithm-specific documentation for tuning tips
- Try adjusting learning rate (usually lower is safer)
- Increase training episodes
- Verify environment is working: `python -c "import gymnasium as gym; gym.make('CartPole-v1').reset()"`

## 📚 Learning Resources

### Recommended Reading Order:
1. Read `docs/REINFORCE_EXPLANATION.md`
2. Read `docs/DQN_EXPLANATION.md`
3. Read `docs/PPO_EXPLANATION.md`
4. Read `docs/DPO_EXPLANATION.md`
5. Read `docs/ALGORITHMS_COMPARISON.md` for overview

### External Resources:
- **Sutton & Barto**: "Reinforcement Learning: An Introduction" (free online)
- **OpenAI Spinning Up**: Excellent RL tutorials
- **DeepMind x UCL**: RL Course videos
- **Stable-Baselines3**: Reference implementations

## 🤝 Contributing

Feel free to:
- Add new algorithms
- Improve existing implementations
- Enhance documentation
- Report bugs or suggest features

## 📝 License

This project is for educational purposes.

## 🙏 Acknowledgments

- OpenAI Gym/Gymnasium for the environment
- Original algorithm authors (Williams, Mnih, Schulman, Rafailov, etc.)
- PyTorch team for the deep learning framework

---

**Happy Learning! 🚀**

For questions or discussions, please refer to the documentation in the `docs/` folder.