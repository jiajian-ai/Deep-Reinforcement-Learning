import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import os
import csv
import matplotlib.pyplot as plt
import pandas as pd


# 创建神经网络策略
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)

    def act(self, state):
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


# REINFORCE算法实现
class REINFORCE:
    def __init__(self, env, state_size, gamma=0.99, lr=0.01):
        self.env = env
        self.state_size = state_size
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        self.policy_net = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.saved_log_probs = []
        self.rewards = []
        self.checkpoint_path = "output/reinforce_checkpoint.pth"
        self.training_history = []
        self.eval_rewards_history = [] # New: To store rewards during evaluation for plotting

    def save_checkpoint(self, episode, total_rewards):
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        torch.save({
            'episode': episode,
            'policy_net_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_rewards': total_rewards,
            'saved_log_probs': self.saved_log_probs,
            'rewards': self.rewards,
        }, self.checkpoint_path)
        print(f"Checkpoint saved at episode {episode}")

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.saved_log_probs = checkpoint['saved_log_probs']
            self.rewards = checkpoint['rewards']
            print(f"Checkpoint loaded from episode {checkpoint['episode']}")
            return checkpoint['episode'], checkpoint['total_rewards']
        return 0, []

    def select_action(self, state):
        action, log_prob = self.policy_net.act(state)
        self.saved_log_probs.append(log_prob)
        return action

    def update_policy(self):
        R = 0
        returns = []

        # 计算每个时间步的未来折扣回报
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)  # 归一化

        policy_loss = []
        for log_prob, G in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * G)

        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

        # 重置存储
        del self.rewards[:]
        del self.saved_log_probs[:]

    def train(self, num_episodes=1000, save_interval=100):
        total_rewards = []
        start_episode, loaded_rewards = self.load_checkpoint()
        total_rewards.extend(loaded_rewards)

        try:
            for episode in range(start_episode, num_episodes):
                state = self.env.reset()[0]
                if isinstance(state, (int, float)): # Check if state is a scalar
                    state = np.array([state] * self.state_size)
                episode_reward = 0

                while True:
                    state = torch.tensor(state).float().unsqueeze(0).to(self.device)
                    action = self.select_action(state)
                    state, reward, done, truncated = self.env.step(action)
                    self.rewards.append(reward)
                    episode_reward += reward

                    if done or truncated:
                        break

                total_rewards.append(episode_reward)
                self.update_policy()

                # 每100轮打印一次训练进度
                if episode % 1 == 0:
                    avg_reward = np.mean(total_rewards[-100:])
                    print(f"Episode {episode}\tAverage Reward (last 100): {avg_reward:.2f}")
                    self.training_history.append({'episode': episode, 'avg_reward': avg_reward})

                # Save checkpoint periodically
                if (episode + 1) % save_interval == 0:
                    self.save_checkpoint(episode + 1, total_rewards)

                # 如果最近100轮平均奖励达到480，认为问题已解决
                if len(total_rewards) >= 100 and np.mean(total_rewards[-100:]) >= 480:
                    print(f"Solved at episode {episode}!")
                    self.save_checkpoint(episode + 1, total_rewards) # Save final checkpoint
                    break

            return total_rewards
        except Exception as e:
            print(f"An error occurred during training: {e}")
            # Save checkpoint before exiting. 'episode' might not be defined if error occurs before loop starts.
            # However, to align with the provided replace_block's logic, we keep 'episode + 1'.
            # A more robust solution would handle 'episode' potentially not being defined.
            self.save_checkpoint(episode + 1, total_rewards) 
            raise # Re-raise the exception after saving

    def save_training_history(self, algorithm_name, scenario_name, output_dir="output"):
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{algorithm_name}_{scenario_name}_training_history.csv"
        filepath = os.path.join(output_dir, filename)
        file_exists = os.path.exists(filepath)
        with open(filepath, 'a', newline='') as csvfile:
            fieldnames = ['episode', 'avg_reward']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write header only once when file is first created
            if not file_exists:
                writer.writeheader()
            for row in self.training_history:
                writer.writerow(row)

def analyze_training_history(algorithm_name, scenario_name, output_dir="output"):
    filepath_csv = os.path.join(output_dir, f"{algorithm_name}_{scenario_name}_training_history.csv")
    filepath_png = os.path.join(output_dir, f"{algorithm_name}_{scenario_name}_training_history.png")

    if not os.path.exists(filepath_csv):
        print(f"Error: Training history file not found at {filepath_csv}")
        return

    df = pd.read_csv(filepath_csv)

    plt.figure(figsize=(10, 6))
    plt.plot(df['episode'], df['avg_reward'])
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("Training History")
    plt.grid(True)
    plt.savefig(filepath_png)
    plt.close()
    print(f"Training history chart saved to {filepath_png}")

def evaluate_and_visualize(agent, env, num_episodes=10, render_interval=5):
    print("\nStarting evaluation with visualization...")
    fig = plt.figure(figsize=(10, 5)) # Adjust figure size for 2x2 layout
    gs = fig.add_gridspec(2, 2, wspace=0.3, hspace=0.3) # 2 rows, 2 columns
    fig.set_constrained_layout(True) # Add this line for better subplot proportions
    
    # Initialize environment display (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('Environment')
    ax1.set_xticks([])
    ax1.set_yticks([])
    img = ax1.imshow(env.render(mode='rgb_array'))
    ax1.set_aspect('equal')

    # Initialize plot for rewards (top-right)
    ax2 = fig.add_subplot(gs[0, 1])
    cumulative_rewards_history = []
    line, = ax2.plot([], [], label='Cumulative Reward')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Cumulative Reward')
    ax2.set_title('Evaluation Cumulative Rewards Over Steps')
    ax2.legend()
    ax2.grid(True)
    ax2.set_xlim(-5, 505) # Fixed x-axis limit
    ax2.set_ylim(0, 510) # Fixed y-axis limit

    # Initialize action history display (bottom, spans both columns)
    ax_action_history = fig.add_subplot(gs[1, :])
    action_history = []
    action_line, = ax_action_history.plot([], [], drawstyle='steps-post', label='Action (0: Left, 1: Right)')
    ax_action_history.set_xlabel('Step')
    ax_action_history.set_ylabel('Action')
    ax_action_history.set_title('Action History (500 steps)')
    ax_action_history.set_xlim(0, 500) # Fixed x-axis for 500 steps
    ax_action_history.set_ylim(-0.1, 1.1) # Actions are 0 or 1
    ax_action_history.set_yticks([0, 1])
    ax_action_history.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax_action_history.legend()

    plt.ion() # Turn on interactive mode
    plt.show(block=False)

    for episode in range(num_episodes):
        state = env.reset()[0]
        episode_cumulative_reward = 0
        done = False
        truncated = False
        step_count = 0
        action_history = [] # Reset action history for each episode

        while not (done or truncated):
            state_np = np.array(state, dtype=np.float32)
            if state_np.ndim == 0:
                state_np = np.repeat(state_np, agent.state_size)
            elif state_np.ndim > 1:
                state_np = state_np.flatten()
            state_tensor = torch.from_numpy(state_np).unsqueeze(0).to(agent.device)
            action, _ = agent.policy_net.act(state_tensor)
            next_state, reward, done, truncated = env.step(action)
            
            state = next_state
            episode_cumulative_reward += reward
            step_count += 1
            # Append reward history every step to keep x/y lengths consistent
            cumulative_rewards_history.append(episode_cumulative_reward)
            action_history.append(action) # Store action history

            if step_count % render_interval == 0:
                # Render environment and update display only at interval
                screen = env.render(mode='rgb_array')
                img.set_array(screen)

                # Update reward plot at interval
                line.set_xdata(range(len(cumulative_rewards_history)))
                line.set_ydata(cumulative_rewards_history)
                ax2.relim()
                ax2.autoscale_view()

                # Update action history plot at interval
                action_line.set_xdata(range(len(action_history)))
                action_line.set_ydata(action_history)
                # ax_action_history.relim() # Not needed with fixed xlim/ylim
                # ax_action_history.autoscale_view() # Not needed with fixed xlim/ylim

                fig.canvas.draw()
                fig.canvas.flush_events()

        print(f"Episode {episode + 1} finished with cumulative reward: {episode_cumulative_reward}")
        # Clear cumulative_rewards_history for the next episode if num_episodes > 1
        if num_episodes > 1:
            cumulative_rewards_history = []

    plt.ioff() # Turn off interactive mode
    env.close()
    plt.show(block=True) # Keep the plot open until closed manually


# 训练和测试模型
if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    agent = REINFORCE(env, state_size, gamma=0.99, lr=0.005)

    # 训练智能体
    print("Starting training...")
    rewards = agent.train(num_episodes=1000, save_interval=50)
    agent.save_training_history(algorithm_name="reinforce", scenario_name="cartpole", output_dir="output")

    # 分析训练历史
    analyze_training_history(algorithm_name="reinforce", scenario_name="cartpole", output_dir="output")

    # 使用可视化函数评估训练好的策略
    evaluate_and_visualize(agent, env, num_episodes=1) # Evaluate for 1 episode with visualization