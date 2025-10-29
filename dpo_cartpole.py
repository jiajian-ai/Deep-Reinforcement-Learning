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
from collections import deque
import random


# 创建策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

    def act(self, state):
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)
    
    def get_log_prob(self, state, action):
        """获取给定状态和动作的对数概率"""
        probs = self.forward(state)
        m = Categorical(probs)
        return m.log_prob(action)


# 轨迹数据存储
class Trajectory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.total_reward = 0
    
    def add(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.total_reward += reward
    
    def __len__(self):
        return len(self.states)


# 偏好数据缓冲区
class PreferenceBuffer:
    def __init__(self, capacity=1000):
        self.buffer = deque(maxlen=capacity)
    
    def add_preference_pair(self, traj_win, traj_lose):
        """添加一对偏好数据：胜出轨迹和失败轨迹"""
        self.buffer.append({
            'win_states': traj_win.states,
            'win_actions': traj_win.actions,
            'win_reward': traj_win.total_reward,
            'lose_states': traj_lose.states,
            'lose_actions': traj_lose.actions,
            'lose_reward': traj_lose.total_reward
        })
    
    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)


# DPO算法实现
class DPO:
    def __init__(self, env, state_size, algorithm_name="dpo", gamma=0.99, lr=0.001, 
                 beta=0.1, batch_size=32, buffer_capacity=1000):
        """
        DPO (Direct Preference Optimization) 算法
        
        参数:
        - beta: DPO的温度参数，控制策略更新的激进程度
        - gamma: 折扣因子
        - lr: 学习率
        - batch_size: 批次大小
        """
        self.env = env
        self.state_size = state_size
        self.gamma = gamma
        self.beta = beta
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        self.action_dim = action_dim

        # 策略网络（要训练的）
        self.policy_net = PolicyNetwork(state_dim, action_dim).to(self.device)
        
        # 参考策略网络（固定的，用于DPO损失计算）
        self.reference_net = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.reference_net.load_state_dict(self.policy_net.state_dict())
        self.reference_net.eval()  # 设为评估模式，不更新
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.preference_buffer = PreferenceBuffer(buffer_capacity)
        
        self.algorithm_output_dir = os.path.join("output", algorithm_name)
        self.checkpoint_path = os.path.join(self.algorithm_output_dir, "dpo_checkpoint.pth")
        self.training_history = []
        self.eval_rewards_history = []

    def save_checkpoint(self, episode, total_rewards):
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        torch.save({
            'episode': episode,
            'policy_net_state_dict': self.policy_net.state_dict(),
            'reference_net_state_dict': self.reference_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_rewards': total_rewards,
        }, self.checkpoint_path)
        print(f"Checkpoint saved at episode {episode}")

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.reference_net.load_state_dict(checkpoint['reference_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Checkpoint loaded from episode {checkpoint['episode']}")
            return checkpoint['episode'], checkpoint['total_rewards']
        return 0, []

    def collect_trajectory(self, use_reference=False):
        """收集一条完整的轨迹"""
        trajectory = Trajectory()
        state = self.env.reset()[0]
        
        if isinstance(state, (int, float)):
            state = np.array([state] * self.state_size)
        
        done = False
        truncated = False
        
        while not (done or truncated):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # 选择使用策略网络或参考网络
            net = self.reference_net if use_reference else self.policy_net
            
            with torch.no_grad():
                action, _ = net.act(state_tensor)
            
            step_result = self.env.step(action)
            
            # 兼容不同版本的gym返回格式
            if len(step_result) == 5:
                next_state, reward, done, truncated, info = step_result
            else:
                next_state, reward, done, info = step_result
                truncated = False
            
            trajectory.add(state, action, reward)
            state = next_state
        
        return trajectory

    def compute_dpo_loss(self, preference_batch):
        """
        计算DPO损失
        
        DPO核心思想：
        最大化胜出轨迹的概率，同时最小化失败轨迹的概率
        损失函数：-log(sigmoid(beta * (log_ratio_win - log_ratio_lose)))
        
        其中 log_ratio = log(π_θ/π_ref)
        """
        total_loss = 0
        
        for pref in preference_batch:
            # 胜出轨迹
            win_states = torch.FloatTensor(np.array(pref['win_states'])).to(self.device)
            win_actions = torch.LongTensor(pref['win_actions']).to(self.device)
            
            # 失败轨迹
            lose_states = torch.FloatTensor(np.array(pref['lose_states'])).to(self.device)
            lose_actions = torch.LongTensor(pref['lose_actions']).to(self.device)
            
            # 计算胜出轨迹的log概率比
            with torch.no_grad():
                win_log_probs_ref = self.reference_net.get_log_prob(win_states, win_actions)
            win_log_probs_policy = self.policy_net.get_log_prob(win_states, win_actions)
            win_log_ratio = (win_log_probs_policy - win_log_probs_ref).sum()
            
            # 计算失败轨迹的log概率比
            with torch.no_grad():
                lose_log_probs_ref = self.reference_net.get_log_prob(lose_states, lose_actions)
            lose_log_probs_policy = self.policy_net.get_log_prob(lose_states, lose_actions)
            lose_log_ratio = (lose_log_probs_policy - lose_log_probs_ref).sum()
            
            # DPO损失：希望胜出轨迹的log_ratio大于失败轨迹的log_ratio
            # 使用sigmoid确保数值稳定性
            logits = self.beta * (win_log_ratio - lose_log_ratio)
            loss = -torch.nn.functional.logsigmoid(logits)
            
            total_loss += loss
        
        return total_loss / len(preference_batch)

    def train(self, num_episodes=1000, save_interval=100, pairs_per_update=2, 
              update_frequency=5, reference_update_freq=50):
        """
        训练DPO智能体
        
        参数:
        - pairs_per_update: 每次收集多少对偏好数据
        - update_frequency: 每收集多少轮进行一次策略更新
        - reference_update_freq: 每多少轮更新一次参考策略
        """
        total_rewards = []
        start_episode, loaded_rewards = self.load_checkpoint()
        total_rewards.extend(loaded_rewards)

        try:
            for episode in range(start_episode, num_episodes):
                # 收集多对轨迹用于偏好比较
                for _ in range(pairs_per_update):
                    traj1 = self.collect_trajectory()
                    traj2 = self.collect_trajectory()
                    
                    # 根据总奖励决定哪个是胜出轨迹
                    if traj1.total_reward > traj2.total_reward:
                        self.preference_buffer.add_preference_pair(traj1, traj2)
                    elif traj2.total_reward > traj1.total_reward:
                        self.preference_buffer.add_preference_pair(traj2, traj1)
                    # 如果奖励相同，不添加（避免无意义的数据）
                
                # 记录当前episode的平均奖励
                avg_episode_reward = (traj1.total_reward + traj2.total_reward) / 2
                total_rewards.append(avg_episode_reward)
                
                # 定期更新策略
                if episode % update_frequency == 0 and len(self.preference_buffer) >= self.batch_size:
                    # 从缓冲区采样并更新策略
                    for _ in range(5):  # 每次更新执行5个梯度步
                        preference_batch = self.preference_buffer.sample(self.batch_size)
                        loss = self.compute_dpo_loss(preference_batch)
                        
                        self.optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
                        self.optimizer.step()
                
                # 定期更新参考策略
                if episode % reference_update_freq == 0 and episode > 0:
                    print(f"Updating reference policy at episode {episode}")
                    self.reference_net.load_state_dict(self.policy_net.state_dict())
                    self.reference_net.eval()

                # 每轮打印一次训练进度
                if episode % 1 == 0:
                    avg_reward = np.mean(total_rewards[-100:])
                    buffer_size = len(self.preference_buffer)
                    print(f"Episode {episode}\tAverage Reward (last 100): {avg_reward:.2f}\tBuffer Size: {buffer_size}")
                    self.training_history.append({'episode': episode, 'avg_reward': avg_reward})

                # Save checkpoint periodically
                if (episode + 1) % save_interval == 0:
                    self.save_checkpoint(episode + 1, total_rewards)

                # 如果最近100轮平均奖励达到480，认为问题已解决
                if len(total_rewards) >= 100 and np.mean(total_rewards[-100:]) >= 480:
                    print(f"Solved at episode {episode}!")
                    self.save_checkpoint(episode + 1, total_rewards)
                    break

            return total_rewards
        except Exception as e:
            print(f"An error occurred during training: {e}")
            self.save_checkpoint(episode + 1, total_rewards)
            raise

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
    plt.title("DPO Training History")
    plt.grid(True)
    plt.savefig(filepath_png)
    plt.close()
    print(f"Training history chart saved to {filepath_png}")


def evaluate_and_visualize(agent, env, num_episodes=10, render_interval=5):
    print("\nStarting evaluation with visualization...")
    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(2, 2, wspace=0.3, hspace=0.3)
    fig.set_constrained_layout(True)
    
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
    ax2.set_xlim(-5, 505)
    ax2.set_ylim(0, 510)

    # Initialize action history display (bottom, spans both columns)
    ax_action_history = fig.add_subplot(gs[1, :])
    action_history = []
    action_line, = ax_action_history.plot([], [], drawstyle='steps-post', label='Action (0: Left, 1: Right)')
    ax_action_history.set_xlabel('Step')
    ax_action_history.set_ylabel('Action')
    ax_action_history.set_title('Action History (500 steps)')
    ax_action_history.set_xlim(0, 500)
    ax_action_history.set_ylim(-0.1, 1.1)
    ax_action_history.set_yticks([0, 1])
    ax_action_history.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax_action_history.legend()

    plt.ion()
    plt.show(block=False)

    for episode in range(num_episodes):
        state = env.reset()[0]
        episode_cumulative_reward = 0
        done = False
        truncated = False
        step_count = 0
        action_history = []

        while not (done or truncated):
            state_np = np.array(state, dtype=np.float32)
            if state_np.ndim == 0:
                state_np = np.repeat(state_np, agent.state_size)
            elif state_np.ndim > 1:
                state_np = state_np.flatten()
            
            # 使用策略网络进行评估
            state_tensor = torch.FloatTensor(state_np).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                action, _ = agent.policy_net.act(state_tensor)
            
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
            cumulative_rewards_history.append(episode_cumulative_reward)
            action_history.append(action)

            if step_count % render_interval == 0:
                screen = env.render(mode='rgb_array')
                img.set_array(screen)

                line.set_xdata(range(len(cumulative_rewards_history)))
                line.set_ydata(cumulative_rewards_history)
                ax2.relim()
                ax2.autoscale_view()

                action_line.set_xdata(range(len(action_history)))
                action_line.set_ydata(action_history)

                fig.canvas.draw()
                fig.canvas.flush_events()

        print(f"Episode {episode + 1} finished with cumulative reward: {episode_cumulative_reward}")
        if num_episodes > 1:
            cumulative_rewards_history = []

    plt.ioff()
    env.close()
    plt.show(block=True)


# 训练和测试模型
if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    algorithm_name = "dpo"
    
    # DPO超参数说明：
    # - beta: 控制策略更新的激进程度，越大越激进
    # - lr: 学习率
    # - batch_size: 每次更新使用的偏好对数量
    agent = DPO(env, state_size, algorithm_name=algorithm_name, 
                gamma=0.99, lr=0.0003, beta=0.5, 
                batch_size=16, buffer_capacity=1000)

    # 训练智能体
    print("Starting DPO training...")
    print("=" * 70)
    print("DPO (Direct Preference Optimization) 算法说明:")
    print("- 通过比较轨迹对来学习，无需显式奖励函数")
    print("- 核心思想：最大化好轨迹相对于差轨迹的概率")
    print("- 使用参考策略防止策略偏离过远")
    print("=" * 70)
    
    rewards = agent.train(num_episodes=1000, save_interval=50, 
                         pairs_per_update=2, update_frequency=5, 
                         reference_update_freq=50)
    agent.save_training_history(algorithm_name=algorithm_name, scenario_name="cartpole")

    # 分析训练历史
    analyze_training_history(algorithm_name=algorithm_name, scenario_name="cartpole", 
                            algorithm_output_dir=agent.algorithm_output_dir)

    # 使用可视化函数评估训练好的策略
    evaluate_and_visualize(agent, env, num_episodes=1)

