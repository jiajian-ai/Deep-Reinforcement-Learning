import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import csv
import matplotlib.pyplot as plt
import pandas as pd
import random
from collections import deque


# 创建神经网络Q网络
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


# DQN算法实现
class DQN:
    def __init__(self, env, state_size, algorithm_name="dqn", gamma=0.99, lr=0.001, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 batch_size=64, buffer_capacity=10000, target_update=10):
        self.env = env
        self.state_size = state_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        self.action_dim = action_dim

        # Q网络和目标Q网络
        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        self.algorithm_output_dir = os.path.join("output", algorithm_name)
        self.checkpoint_path = os.path.join(self.algorithm_output_dir, "dqn_checkpoint.pth")
        self.training_history = []
        self.eval_rewards_history = []

    def save_checkpoint(self, episode, total_rewards):
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        torch.save({
            'episode': episode,
            'q_net_state_dict': self.q_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_rewards': total_rewards,
            'epsilon': self.epsilon,
        }, self.checkpoint_path)
        print(f"Checkpoint saved at episode {episode}")

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path)
            self.q_net.load_state_dict(checkpoint['q_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            print(f"Checkpoint loaded from episode {checkpoint['episode']}")
            return checkpoint['episode'], checkpoint['total_rewards']
        return 0, []

    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_net(state)
                return q_values.argmax(1).item()

    def update_q_network(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # 转换为张量
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # 计算当前Q值
        current_q_values = self.q_net(states).gather(1, actions)
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # 计算损失并更新
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, num_episodes=1000, save_interval=100):
        total_rewards = []
        start_episode, loaded_rewards = self.load_checkpoint()
        total_rewards.extend(loaded_rewards)

        try:
            for episode in range(start_episode, num_episodes):
                state = self.env.reset()[0]
                if isinstance(state, (int, float)):  # Check if state is a scalar
                    state = np.array([state] * self.state_size)
                episode_reward = 0

                while True:
                    action = self.select_action(state, training=True)
                    step_result = self.env.step(action)
                    
                    # 兼容不同版本的gym返回格式
                    if len(step_result) == 5:
                        next_state, reward, done, truncated, info = step_result
                    else:
                        next_state, reward, done, info = step_result
                        truncated = False
                    
                    # 存储经验
                    self.replay_buffer.push(state, action, reward, next_state, done or truncated)
                    
                    state = next_state
                    episode_reward += reward
                    
                    # 更新Q网络
                    self.update_q_network()

                    if done or truncated:
                        break

                total_rewards.append(episode_reward)
                
                # 衰减epsilon
                if self.epsilon > self.epsilon_end:
                    self.epsilon *= self.epsilon_decay

                # 定期更新目标网络
                if episode % self.target_update == 0:
                    self.target_net.load_state_dict(self.q_net.state_dict())

                # 每轮打印一次训练进度
                if episode % 1 == 0:
                    avg_reward = np.mean(total_rewards[-100:])
                    print(f"Episode {episode}\tAverage Reward (last 100): {avg_reward:.2f}\tEpsilon: {self.epsilon:.3f}")
                    self.training_history.append({'episode': episode, 'avg_reward': avg_reward})

                # Save checkpoint periodically
                if (episode + 1) % save_interval == 0:
                    self.save_checkpoint(episode + 1, total_rewards)

                # 如果最近100轮平均奖励达到480，认为问题已解决
                if len(total_rewards) >= 100 and np.mean(total_rewards[-100:]) >= 480:
                    print(f"Solved at episode {episode}!")
                    self.save_checkpoint(episode + 1, total_rewards)  # Save final checkpoint
                    break

            return total_rewards
        except Exception as e:
            print(f"An error occurred during training: {e}")
            self.save_checkpoint(episode + 1, total_rewards)
            raise  # Re-raise the exception after saving

    def save_training_history(self, algorithm_name, scenario_name):
        os.makedirs(self.algorithm_output_dir, exist_ok=True)
        filename = f"{algorithm_name}_{scenario_name}_training_history.csv"
        filepath = os.path.join(self.algorithm_output_dir, filename)
        file_exists = os.path.exists(filepath)
        with open(filepath, 'a', newline='') as csvfile:
            fieldnames = ['episode', 'avg_reward']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write header only once when file is first created
            if not file_exists:
                writer.writeheader()
            for row in self.training_history:
                writer.writerow(row)


def analyze_training_history(algorithm_name, scenario_name, algorithm_output_dir):
    filepath_csv = os.path.join(algorithm_output_dir, f"{algorithm_name}_{scenario_name}_training_history.csv")
    filepath_png = os.path.join(algorithm_output_dir, f"{algorithm_name}_{scenario_name}_training_history.png")

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
    fig = plt.figure(figsize=(10, 5))  # Adjust figure size for 2x2 layout
    gs = fig.add_gridspec(2, 2, wspace=0.3, hspace=0.3)  # 2 rows, 2 columns
    fig.set_constrained_layout(True)  # Add this line for better subplot proportions
    
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
    ax2.set_xlim(-5, 505)  # Fixed x-axis limit
    ax2.set_ylim(0, 510)  # Fixed y-axis limit

    # Initialize action history display (bottom, spans both columns)
    ax_action_history = fig.add_subplot(gs[1, :])
    action_history = []
    action_line, = ax_action_history.plot([], [], drawstyle='steps-post', label='Action (0: Left, 1: Right)')
    ax_action_history.set_xlabel('Step')
    ax_action_history.set_ylabel('Action')
    ax_action_history.set_title('Action History (500 steps)')
    ax_action_history.set_xlim(0, 500)  # Fixed x-axis for 500 steps
    ax_action_history.set_ylim(-0.1, 1.1)  # Actions are 0 or 1
    ax_action_history.set_yticks([0, 1])
    ax_action_history.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax_action_history.legend()

    plt.ion()  # Turn on interactive mode
    plt.show(block=False)

    for episode in range(num_episodes):
        state = env.reset()[0]
        episode_cumulative_reward = 0
        done = False
        truncated = False
        step_count = 0
        action_history = []  # Reset action history for each episode

        while not (done or truncated):
            state_np = np.array(state, dtype=np.float32)
            if state_np.ndim == 0:
                state_np = np.repeat(state_np, agent.state_size)
            elif state_np.ndim > 1:
                state_np = state_np.flatten()
            
            # 使用贪婪策略进行评估（不探索）
            action = agent.select_action(state_np, training=False)
            step_result = env.step(action)
            
            # 兼容不同版本的gym返回格式
            if len(step_result) == 5:
                next_state, reward, done, truncated, info = step_result
            else:
                next_state, reward, done, info = step_result
                truncated = False
            
            state = next_state
            episode_cumulative_reward += reward
            step_count += 1
            # Append reward history every step to keep x/y lengths consistent
            cumulative_rewards_history.append(episode_cumulative_reward)
            action_history.append(action)  # Store action history

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

                fig.canvas.draw()
                fig.canvas.flush_events()

        print(f"Episode {episode + 1} finished with cumulative reward: {episode_cumulative_reward}")
        # Clear cumulative_rewards_history for the next episode if num_episodes > 1
        if num_episodes > 1:
            cumulative_rewards_history = []

    plt.ioff()  # Turn off interactive mode
    env.close()
    plt.show(block=True)  # Keep the plot open until closed manually


# 训练和测试模型
if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    algorithm_name = "dqn"
    agent = DQN(env, state_size, algorithm_name=algorithm_name, gamma=0.99, lr=0.001,
                epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                batch_size=64, buffer_capacity=10000, target_update=10)

    # 训练智能体
    print("Starting training...")
    rewards = agent.train(num_episodes=1000, save_interval=50)
    agent.save_training_history(algorithm_name=algorithm_name, scenario_name="cartpole")

    # 分析训练历史
    analyze_training_history(algorithm_name=algorithm_name, scenario_name="cartpole", algorithm_output_dir=agent.algorithm_output_dir)

    # 使用可视化函数评估训练好的策略
    evaluate_and_visualize(agent, env, num_episodes=1)  # Evaluate for 1 episode with visualization

