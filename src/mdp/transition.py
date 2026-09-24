import torch
import torch.nn.functional as F


class TransitionEstimator:
    """转移概率估计目标的处理工具。

    核心思想：将概率校准定义为状态转移概率估计问题。
    模型需要估计 P(s' | s, a)，即在状态 s 下采取动作 a 后转移到
    各可能后续状态的概率分布。
    """

    @staticmethod
    def compute_kl_divergence(predicted, target, eps=1e-8):
        """计算预测分布与目标分布之间的 KL 散度。

        Args:
            predicted: (batch, num_actions, num_outcomes) 预测转移概率
            target: (batch, num_actions, num_outcomes) 目标转移概率
            eps: 数值稳定性小量

        Returns:
            kl: (batch, num_actions) KL 散度
        """
        predicted = torch.clamp(predicted, min=eps, max=1.0 - eps)
        target = torch.clamp(target, min=eps, max=1.0 - eps)
        predicted = predicted / predicted.sum(dim=-1, keepdim=True)
        target = target / target.sum(dim=-1, keepdim=True)
        kl = torch.sum(target * torch.log(target / predicted), dim=-1)
        return kl

    @staticmethod
    def compute_brier_score(predicted, target):
        """计算 Brier Score（概率预测准确度）。

        Brier Score = (1/N) * sum(predicted - target)^2

        Args:
            predicted: (batch, num_actions, num_outcomes)
            target: (batch, num_actions, num_outcomes)

        Returns:
            brier: (batch, num_actions) Brier Score
        """
        return torch.sum((predicted - target) ** 2, dim=-1)

    @staticmethod
    def compute_confidence_calibration(predicted, targets):
        """计算置信度校准误差。

        Args:
            predicted: (batch, num_actions, num_outcomes) 预测概率
            targets: (batch, num_actions, num_outcomes) 真实分布（通常是 one-hot）

        Returns:
            ece: 期望校准误差 (Expected Calibration Error)
            mce: 最大校准误差 (Maximum Calibration Error)
        """
        batch_size, num_actions, num_outcomes = predicted.shape
        device = predicted.device

        predicted_probs = predicted.reshape(-1, num_outcomes)
        target_labels = targets.reshape(-1, num_outcomes)

        confidences, predictions = torch.max(predicted_probs, dim=1)
        correct = (predictions == torch.argmax(target_labels, dim=1)).float()

        num_bins = 10
        bin_boundaries = torch.linspace(0, 1, num_bins + 1, device=device)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = torch.zeros(1, device=device)
        mce = torch.zeros(1, device=device)

        for i in range(num_bins):
            in_bin = (confidences > bin_lowers[i]) & (confidences <= bin_uppers[i])
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin > 0:
                avg_confidence = confidences[in_bin].mean()
                avg_accuracy = correct[in_bin].mean()
                bin_ece = torch.abs(avg_confidence - avg_accuracy)
                ece = ece + prop_in_bin * bin_ece
                mce = torch.max(mce, bin_ece.unsqueeze(0))

        return ece, mce

    @staticmethod
    def normalize_probabilities(logits):
        """将 logits 归一化为概率分布。

        Args:
            logits: (batch, num_actions, num_outcomes)

        Returns:
            probs: (batch, num_actions, num_outcomes) 归一化概率
        """
        return F.softmax(logits, dim=-1)
