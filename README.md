# Deep Reinforcement Learning for CartPole

This project implements a Deep Reinforcement Learning (DRL) agent using the REINFORCE algorithm to solve the CartPole-v1 environment from OpenAI Gym.

## Project Structure

- `reinforce_cartpole.py`: The main script containing the DRL agent, training, evaluation, and visualization logic.
- `output/`: Directory to store training checkpoints, history, and visualization outputs.

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

To train and evaluate the REINFORCE agent, run the main script:

```bash
python reinforce_cartpole.py
```

The script will:
- Initialize the CartPole environment and the REINFORCE agent.
- Train the agent, saving checkpoints periodically.
- Save the training history and a plot of cumulative rewards.
- Evaluate the trained agent with real-time visualization of the environment, cumulative rewards, and action history.

## Visualization

The `evaluate_and_visualize` function displays a 2x2 grid with the following subplots:

- **Top-Left:** Real-time rendering of the CartPole environment.
- **Top-Right:** Cumulative reward curve over evaluation steps.
- **Bottom (spanning both columns):** A square wave representation of the agent's actions (Left/Right) over the last 500 steps.

You can adjust the `render_interval` parameter in the `evaluate_and_visualize` function to control the visualization speed.

## Checkpoints and Output

- Training checkpoints are saved in the `output/` directory as `reinforce_checkpoint.pth`.
- Training history (CSV) and a plot of the training history (PNG) are saved in the `output/` directory.

## Future Improvements (Optional)

- Implement other DRL algorithms (e.g., A2C, PPO).
- Add more sophisticated logging and monitoring.
- Explore different hyperparameter tuning strategies.
- Save evaluation visualizations as video or GIF.