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
import argparse


# Actor 网络（策略）
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
        return action.item(), m.log_prob(action), probs


# Critic 网络（价值）
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_size=128):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class PPO:
    def __init__(self, env, state_size, algorithm_name="ppo", gamma=0.99,
                 lr_actor=1e-2, lr_critic=1e-2, clip_epsilon=0.2, train_epochs=20,
                 gae_lambda=0.95, rollout_steps=2048, batch_size=128,
                 entropy_coef=0.01, value_coef=0.5, max_grad_norm=0.5, resume=True, hidden_size=128):
        self.env = env
        self.state_size = state_size
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.train_epochs = train_epochs
        self.gae_lambda = gae_lambda
        self.rollout_steps = rollout_steps
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.resume = resume # Add this line
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_size).to(self.device)
        self.value_net = ValueNetwork(state_dim, hidden_size).to(self.device)
        self.optimizer_actor = optim.Adam(self.policy_net.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.value_net.parameters(), lr=lr_critic)

        # 轨迹缓存
        self.memory = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'dones': [],
            'values': [],
        }

        self.algorithm_output_dir = os.path.join("output", algorithm_name)
        self.checkpoint_path = os.path.join(self.algorithm_output_dir, "ppo_checkpoint.pth")
        self.training_history = []

    def save_checkpoint(self, episode, total_rewards):
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        torch.save({
            'episode': episode,
            'policy_net_state_dict': self.policy_net.state_dict(),
            'value_net_state_dict': self.value_net.state_dict(),
            'optimizer_actor_state_dict': self.optimizer_actor.state_dict(),
            'optimizer_critic_state_dict': self.optimizer_critic.state_dict(),
            'total_rewards': total_rewards,
        }, self.checkpoint_path)
        print(f"Checkpoint saved at episode {episode}")

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
            self.optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state_dict'])
            self.optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])
            print(f"Checkpoint loaded from episode {checkpoint['episode']}")
            return checkpoint['episode'], checkpoint['total_rewards']
        return 0, []

    def select_action(self, state_tensor):
        action, log_prob, probs = self.policy_net.act(state_tensor)
        value = self.value_net(state_tensor).squeeze(-1).squeeze(0)
        return action, log_prob, value, probs

    def step_env(self, action):
        """Compatibility wrapper for Gym step outputs across versions.
        Ensures we always return: (next_state, reward, done, truncated).
        """
        result = self.env.step(action)
        if isinstance(result, tuple):
            if len(result) == 5:
                next_state, reward, terminated, truncated, info = result
                done = bool(terminated)
                return next_state, reward, done, bool(truncated)
            elif len(result) == 4:
                next_state, reward, done, info = result
                # Older Gym returns 4-tuple; no explicit truncated flag
                return next_state, reward, bool(done), False
        # Fallback: try 3-tuple (very old API)
        try:
            next_state, reward, done = result
            return next_state, reward, bool(done), False
        except Exception:
            raise ValueError(f"Unexpected env.step() return format: {type(result)} / len unknown")

    def _compute_gae(self, last_value):
        # Generalized Advantage Estimation
        rewards = self.memory['rewards']
        dones = self.memory['dones']
        values = [v.detach() for v in self.memory['values']]
        values = values + [last_value.detach()]

        advantages = []
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1.0 - float(dones[t])) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - float(dones[t])) * gae
            advantages.insert(0, gae)

        advantages = torch.stack(advantages).to(self.device)
        returns = advantages + torch.stack(self.memory['values']).detach()

        # Normalize advantages for stable updates
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def update_policy(self, last_value):
        states = torch.stack(self.memory['states']).detach()
        actions = torch.tensor(self.memory['actions'], dtype=torch.int64, device=self.device)
        old_log_probs = torch.stack(self.memory['log_probs']).detach()

        returns, advantages = self._compute_gae(last_value)

        dataset_size = states.size(0)
        indices = torch.randperm(dataset_size, device=self.device)

        for _ in range(self.train_epochs):
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                mb_idx = indices[start:end]

                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                probs = self.policy_net(mb_states)
                dist = Categorical(probs)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = (new_log_probs - mb_old_log_probs).exp()
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * mb_advantages
                actor_loss = -(torch.min(surr1, surr2)).mean() - self.entropy_coef * entropy

                values_pred = self.value_net(mb_states).squeeze(-1)
                critic_loss = nn.MSELoss()(values_pred, mb_returns)

                self.optimizer_actor.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                self.optimizer_actor.step()

                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                self.optimizer_critic.step()

        # 清空缓存
        for k in self.memory:
            self.memory[k] = []

    def train(self, num_episodes=1000, save_interval=50):
        total_rewards = []
        start_episode = 0
        loaded_rewards = []
        if self.resume:
            start_episode, loaded_rewards = self.load_checkpoint()
        total_rewards.extend(loaded_rewards)

        try:
            for episode in range(start_episode, num_episodes):
                state = self.env.reset()[0]
                if isinstance(state, (int, float)):
                    state = np.array([state] * self.state_size)
                episode_reward = 0

                done = False
                truncated = False
                while not (done or truncated):
                    state_np = np.array(state, dtype=np.float32)
                    if state_np.ndim == 0:
                        state_np = np.repeat(state_np, self.state_size)
                    elif state_np.ndim > 1:
                        state_np = state_np.flatten()
                    state_tensor = torch.from_numpy(state_np).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        action, log_prob, value, _ = self.select_action(state_tensor)
                    next_state, reward, done, truncated = self.step_env(action)

                    # 存储轨迹
                    self.memory['states'].append(state_tensor.squeeze(0))
                    self.memory['actions'].append(action)
                    self.memory['log_probs'].append(log_prob)
                    self.memory['rewards'].append(reward)
                    self.memory['dones'].append(done or truncated)
                    self.memory['values'].append(value)

                    state = next_state
                    episode_reward += reward

                total_rewards.append(episode_reward)

                # 如果收集到足够的步数，则进行一次更新
                if len(self.memory['states']) >= self.rollout_steps:
                    if done or truncated:
                        last_value = torch.tensor(0.0, device=self.device)
                    else:
                        state_np = np.array(state, dtype=np.float32)
                        if state_np.ndim == 0:
                            state_np = np.repeat(state_np, self.state_size)
                        elif state_np.ndim > 1:
                            state_np = state_np.flatten()
                        state_tensor = torch.from_numpy(state_np).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            last_value = self.value_net(state_tensor).squeeze(-1).squeeze(0)
                    self.update_policy(last_value)

                avg_reward = np.mean(total_rewards[-100:])
                print(f"Episode {episode}\tAverage Reward (last 100): {avg_reward:.2f}")
                self.training_history.append({'episode': episode, 'avg_reward': avg_reward})

                if (episode + 1) % save_interval == 0:
                    self.save_checkpoint(episode + 1, total_rewards)

                if len(total_rewards) >= 100 and np.mean(total_rewards[-100:]) >= 480:
                    print(f"Solved at episode {episode}!")
                    self.save_checkpoint(episode + 1, total_rewards)
                    break

            # Final update if there are remaining samples in memory
            if len(self.memory['states']) > 0:
                # Bootstrap last value if episode did not end (safety, though typically episodes end)
                if done or truncated:
                    last_value = torch.tensor(0.0, device=self.device)
                else:
                    state_np = np.array(state, dtype=np.float32)
                    if state_np.ndim == 0:
                        state_np = np.repeat(state_np, self.state_size)
                    elif state_np.ndim > 1:
                        state_np = state_np.flatten()
                    state_tensor = torch.from_numpy(state_np).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        last_value = self.value_net(state_tensor).squeeze(-1).squeeze(0)
                self.update_policy(last_value)

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


def evaluate_and_visualize(agent, env, num_episodes=1, render_interval=5):
    print("\nStarting evaluation with visualization...")
    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(2, 2, wspace=0.3, hspace=0.3)
    fig.set_constrained_layout(True)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('Environment')
    ax1.set_xticks([])
    ax1.set_yticks([])
    img = ax1.imshow(env.render(mode='rgb_array'))
    ax1.set_aspect('equal')

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
            state_tensor = torch.from_numpy(state_np).unsqueeze(0).to(agent.device)
            action, log_prob, _value, _probs = agent.policy_net.act(state_tensor)
            next_state, reward, done, truncated = env.step(action)

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


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    algorithm_name = "ppo"

    parser = argparse.ArgumentParser(description='PPO training for CartPole-v1')
    parser.add_argument('--resume', action='store_true', help='Resume training from a checkpoint (default: True)')
    parser.add_argument('--no-resume', dest='resume', action='store_false', help='Start training from scratch (default: False)')
    parser.add_argument('--hidden-size', type=int, default=512, help='Hidden layer size for policy and value networks (default: 256)')
    parser.set_defaults(resume=True)
    args = parser.parse_args()

    # Use improved defaults defined in PPO for more stable training
    agent = PPO(env, state_size, algorithm_name=algorithm_name, resume=args.resume, hidden_size=args.hidden_size)

    print("Starting PPO training...")
    rewards = agent.train(num_episodes=1000, save_interval=50)
    agent.save_training_history(algorithm_name=algorithm_name, scenario_name="cartpole")

    analyze_training_history(algorithm_name=algorithm_name, scenario_name="cartpole", algorithm_output_dir=agent.algorithm_output_dir)

    # evaluate_and_visualize(agent, env, num_episodes=1)