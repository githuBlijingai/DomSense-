import torch


class RewardFunction:
    """奖励函数定义。

    奖励由两部分组成：
    1. 决策正确奖励 (decision_correct): 模型选择的动作与真实最优动作一致
    2. 转移准确奖励 (transition_accuracy): 模型的转移概率估计与真实分布接近
    """

    def __init__(self, decision_weight=1.0, transition_weight=1.0):
        self.decision_weight = decision_weight
        self.transition_weight = transition_weight

    def compute_reward(self, selected_actions, target_actions, predicted_transitions, target_transitions):
        """计算总奖励。

        Args:
            selected_actions: (batch,) 模型选择的动作索引
            target_actions: (batch,) 真实最优动作
            predicted_transitions: (batch, num_actions, num_outcomes) 预测转移概率
            target_transitions: (batch, num_actions, num_outcomes) 真实转移概率

        Returns:
            rewards: (batch, num_actions) 每个动作的奖励
        """
        batch_size = selected_actions.shape[0]
        num_actions = predicted_transitions.shape[1]
        device = selected_actions.device

        rewards = torch.zeros(batch_size, num_actions, device=device)

        decision_correct = (selected_actions == target_actions).float()
        rewards[:, selected_actions] += self.decision_weight * decision_correct

        transition_diff = torch.sum(
            (predicted_transitions - target_transitions) ** 2, dim=-1
        )
        transition_accuracy_reward = -self.transition_weight * transition_diff
        rewards = rewards + transition_accuracy_reward

        return rewards

    def compute_sparse_reward(self, selected_actions, target_actions):
        """稀疏奖励：仅决策正确与否。

        Args:
            selected_actions: (batch,) 模型选择的动作
            target_actions: (batch,) 真实最优动作

        Returns:
            reward: (batch,) 标量奖励
        """
        return (selected_actions == target_actions).float()

    def compute_dense_reward(self, q_values, target_q_values):
        """稠密奖励：Q 值预测的负均方误差。

        Args:
            q_values: (batch, num_actions) 预测 Q 值
            target_q_values: (batch, num_actions) 真实 Q 值

        Returns:
            reward: (batch,) 稠密奖励
        """
        return -F.mse_loss(q_values, target_q_values, reduction="none").mean(dim=1)
