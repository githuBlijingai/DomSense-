import torch
import torch.nn.functional as F


class MDP:
    """马尔可夫决策过程 (MDP) 数据结构和基本运算。

    M = (S, A, P, R, gamma)

    在这个模型中的定义：
    - S (状态): 用户输入，由 LLM 编码器编码为语义嵌入
    - A (动作): 预定义的候选选项（1-3个）
    - P (转移概率): P(s'|s,a)，由 Transition Head 估计
    - R (奖励): 决策正确 + 转移准确的奖励信号
    - gamma (折扣因子): 对未来回报的折扣系数
    """

    def __init__(self, num_actions=3, num_outcomes=2, gamma=0.9):
        self.num_actions = num_actions
        self.num_outcomes = num_outcomes
        self.gamma = gamma

    def compute_expected_return(self, reward, transition_probs, next_q_values):
        """计算给定动作的期望回报 Q(s,a)。

        Q(s,a) = R(s,a) + gamma * sum_{s'} P(s'|s,a) * max_{a'} Q(s',a')
        """
        expected_future = torch.sum(transition_probs * next_q_values, dim=-1)
        q_value = reward + self.gamma * expected_future
        return q_value


class BellmanOperator:
    """Bellman 最优性算子，用于隐式多步前瞻。

    在潜空间执行 n 步 Bellman 迭代：
        Q_{k+1}(s,a) = R(s,a) + gamma * sum_{s'} P(s'|s,a) * max_{a'} Q_k(s',a')
    """

    def __init__(self, gamma=0.9, num_steps=4):
        self.gamma = gamma
        self.num_steps = num_steps

    def iterate(self, q_values, rewards, transition_probs):
        """执行多步 Bellman 迭代。

        对每个动作在每个结局下的 Q 值取最大值作为该动作的当前估计，
        再在潜空间/概率空间进行迭代更新。

        Args:
            q_values: (batch, num_actions, num_outcomes) 初始 Q 值
                     (或 (batch, num_actions) 已合并的动作估计)
            rewards: (batch, num_actions, 1) 奖励
            transition_probs: (batch, num_actions, num_outcomes) 转移概率

        Returns:
            q_values: (batch, num_actions) 迭代后的 Q 值
            q_history: list of (batch, num_actions) Q 值历史
        """
        batch_size = q_values.shape[0]
        device = q_values.device

        if q_values.dim() == 3:
            q, _ = torch.max(q_values, dim=-1)
            q = q.squeeze(-1)
        else:
            q = q_values.clone()

        q_history = [q.clone()]

        for step in range(self.num_steps):
            max_next_q = q.max(dim=-1).values
            expected_future = torch.sum(transition_probs * max_next_q.unsqueeze(1).unsqueeze(-1), dim=-1)
            q = rewards.squeeze(-1) + self.gamma * expected_future
            q_history.append(q.clone())

        return q, q_history

    def value_iteration(self, q_values, rewards, transition_probs, tol=1e-6, max_iter=100):
        """值迭代直到收敛（用于测试验证）。

        Args:
            q_values: (batch, num_actions, num_outcomes)
            rewards: (batch, num_actions, 1)
            transition_probs: (batch, num_actions, num_outcomes)
            tol: 收敛容忍度
            max_iter: 最大迭代次数

        Returns:
            q_values: (batch, num_actions) 收敛后的 Q 值
            n_iters: 实际迭代次数
        """
        q = q_values.clone()

        for i in range(max_iter):
            q_prev = q.clone()
            max_next_q, _ = torch.max(q, dim=1, keepdim=True)
            max_next_q = max_next_q.squeeze(1)
            expected_future = torch.sum(transition_probs * max_next_q.unsqueeze(1), dim=-1)
            q = rewards.squeeze(-1) + self.gamma * expected_future
            diff = torch.max(torch.abs(q - q_prev)).item()
            if diff < tol:
                return q, i + 1

        return q, max_iter
