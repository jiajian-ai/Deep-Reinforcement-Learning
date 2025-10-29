# DPO (Direct Preference Optimization) 算法详解

## 📚 什么是DPO？

DPO（Direct Preference Optimization，直接偏好优化）是一种创新的强化学习算法，由Rafailov等人在2023年提出。它是RLHF（Reinforcement Learning from Human Feedback）的一种简化替代方案。

## 🎯 核心思想

### 传统RLHF的问题
传统的RLHF需要三个步骤：
1. 预训练一个策略模型
2. 训练一个奖励模型（需要大量人类偏好数据）
3. 使用PPO等算法优化策略

这个过程复杂、不稳定，且计算成本高。

### DPO的创新
DPO **直接**从偏好数据优化策略，跳过了奖励模型的训练，将三步简化为一步！

## 🔬 算法原理

### 1. 偏好数据
DPO使用成对的轨迹比较：
- **胜出轨迹** (y_w): 表现更好的轨迹
- **失败轨迹** (y_l): 表现较差的轨迹

例如在CartPole中：
- 轨迹A获得奖励200 → 胜出轨迹
- 轨迹B获得奖励150 → 失败轨迹

### 2. DPO损失函数

```
L_DPO = -log(σ(β * (log_ratio_win - log_ratio_lose)))
```

其中：
- `log_ratio = log(π_θ(a|s) / π_ref(a|s))` 
- `π_θ`: 当前策略
- `π_ref`: 参考策略（防止策略变化过大）
- `β`: 温度参数，控制更新激进程度
- `σ`: sigmoid函数

### 3. 直观理解

DPO做的事情很简单：
1. 看两条轨迹，一条好一条差
2. **增加**好轨迹中动作的概率
3. **降低**差轨迹中动作的概率
4. 但不要偏离参考策略太远（通过log_ratio约束）

## 💻 代码实现要点

### 1. 轨迹收集
```python
def collect_trajectory(self):
    trajectory = Trajectory()
    while not done:
        action = policy.act(state)
        next_state, reward = env.step(action)
        trajectory.add(state, action, reward)
    return trajectory
```

### 2. 偏好对生成
```python
traj1 = collect_trajectory()
traj2 = collect_trajectory()

if traj1.total_reward > traj2.total_reward:
    buffer.add_preference_pair(traj1, traj2)  # traj1是胜出者
else:
    buffer.add_preference_pair(traj2, traj1)  # traj2是胜出者
```

### 3. DPO损失计算
```python
def compute_dpo_loss(win_traj, lose_traj):
    # 计算胜出轨迹的log概率比
    win_log_ratio = sum(log(π_θ/π_ref) for win_traj)
    
    # 计算失败轨迹的log概率比
    lose_log_ratio = sum(log(π_θ/π_ref) for lose_traj)
    
    # DPO损失
    logits = beta * (win_log_ratio - lose_log_ratio)
    loss = -log_sigmoid(logits)
    
    return loss
```

## 🔑 关键超参数

### 1. beta (温度参数)
- **作用**: 控制策略更新的激进程度
- **典型值**: 0.1 - 1.0
- **大值**: 更激进的更新，可能不稳定
- **小值**: 更保守的更新，收敛慢

### 2. reference_update_freq (参考策略更新频率)
- **作用**: 多久更新一次参考策略
- **典型值**: 每50-100个episode
- **意义**: 让参考策略跟上训练进度

### 3. pairs_per_update (每次更新的偏好对数量)
- **作用**: 每轮收集多少对轨迹
- **典型值**: 2-5对
- **权衡**: 更多对→数据更丰富，但收集更慢

## 📊 DPO vs 其他算法

| 算法 | 需要奖励模型 | 训练稳定性 | 计算效率 | 样本效率 |
|------|-------------|-----------|---------|---------|
| PPO  | 否 | 中 | 中 | 中 |
| DQN  | 否 | 高 | 高 | 低 |
| REINFORCE | 否 | 低 | 高 | 低 |
| DPO  | 否 | 高 | 高 | 高 |
| RLHF | 是 | 低 | 低 | 高 |

## 🎓 DPO的优势

1. **简单**: 不需要训练奖励模型
2. **稳定**: 比PPO等算法更稳定
3. **高效**: 直接优化偏好，样本效率高
4. **灵活**: 可以用于任何有偏好数据的场景

## 🚀 DPO的应用

### 1. 大语言模型对齐
- ChatGPT、Claude等模型的人类偏好对齐
- 替代复杂的RLHF流程

### 2. 机器人控制
- 从示范轨迹学习最优策略
- 不需要精确的奖励函数

### 3. 游戏AI
- 从专家对局学习
- 比传统RL更高效

## 📈 训练技巧

### 1. 偏好数据质量
- 确保胜出/失败轨迹有明显差异
- 避免奖励相同的轨迹对（无信息）

### 2. 参考策略更新
- 太频繁: 策略变化不明显
- 太慢: 可能导致过度优化

### 3. 批次大小
- 小批次: 更新更频繁，但可能不稳定
- 大批次: 更新更稳定，但需要更多数据

## 🔬 实验建议

### 初次尝试
```python
agent = DPO(
    beta=0.5,           # 中等激进程度
    lr=0.0003,          # 较小的学习率
    batch_size=16,      # 中等批次
    pairs_per_update=2  # 每轮2对轨迹
)
```

### 如果训练不稳定
- 降低 `beta` (0.1 - 0.3)
- 降低 `lr` (0.0001)
- 增加 `reference_update_freq`

### 如果收敛太慢
- 增加 `beta` (0.7 - 1.0)
- 增加 `pairs_per_update` (3-5)
- 增加 `batch_size` (32-64)

## 📚 参考文献

1. Rafailov, R., et al. (2023). "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
2. Christiano, P., et al. (2017). "Deep Reinforcement Learning from Human Preferences"

## 🎯 学习路径

1. **理解基础**: 先掌握策略梯度方法（REINFORCE）
2. **运行代码**: 运行 `dpo_cartpole.py` 观察训练过程
3. **调整参数**: 尝试不同的 `beta` 值，观察影响
4. **对比实验**: 与PPO、DQN对比性能
5. **深入研究**: 阅读原论文，理解理论推导

## 💡 总结

DPO是一个**简单但强大**的算法，它通过直接优化偏好数据，避免了复杂的奖励建模。在很多场景下，它比传统RLHF更高效、更稳定。

**核心记忆点**:
- 用偏好对代替奖励函数
- 直接优化策略，无需中间奖励模型
- 通过参考策略保持稳定性

