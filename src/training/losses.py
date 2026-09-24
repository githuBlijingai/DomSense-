import torch
import torch.nn as nn
import torch.nn.functional as F


class CalibrationLoss(nn.Module):
    """校准损失：衡量转移概率预测与真实分布之间的差距。

    核心思想：概率校准 = 状态转移概率估计。
    使用 KL 散度和 Brier Score 作为校准度量。
    """

    def __init__(self, method="kl", eps=1e-8):
        super().__init__()
        self.method = method
        self.eps = eps

    def forward(self, predicted_probs, target_probs, valid_mask=None):
        """计算校准损失。

        Args:
            predicted_probs: (batch, num_actions, num_outcomes) 预测转移概率
            target_probs: (batch, num_actions, num_outcomes) 真实转移概率
            valid_mask: (batch, num_actions) 有效动作 mask

        Returns:
            loss: 标量校准损失
        """
        predicted = torch.clamp(predicted_probs, min=self.eps, max=1.0 - self.eps)
        target = torch.clamp(target_probs, min=self.eps, max=1.0 - self.eps)

        if self.method == "kl":
            predicted = predicted / predicted.sum(dim=-1, keepdim=True)
            target = target / target.sum(dim=-1, keepdim=True)
            loss = torch.sum(target * torch.log(target / predicted), dim=-1)
        elif self.method == "brier":
            loss = torch.sum((predicted - target) ** 2, dim=-1)
        elif self.method == "cross_entropy":
            loss = -torch.sum(target * torch.log(predicted), dim=-1)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        if valid_mask is not None:
            loss = loss * valid_mask.float()
            return loss.sum() / valid_mask.float().sum().clamp(min=1)

        return loss.mean()


class QValueLoss(nn.Module):
    """Q 值损失：衡量期望回报预测的准确性。"""

    def __init__(self, method="mse"):
        super().__init__()
        self.method = method

    def forward(self, predicted_q, target_q, valid_mask=None):
        """计算 Q 值预测损失。

        Args:
            predicted_q: (batch, num_actions) 预测 Q 值
            target_q: (batch, num_actions) 真实 Q 值
            valid_mask: (batch, num_actions) 有效动作 mask

        Returns:
            loss: 标量损失
        """
        if self.method == "mse":
            loss = F.mse_loss(predicted_q, target_q, reduction="none")
        elif self.method == "huber":
            loss = F.huber_loss(predicted_q, target_q, reduction="none")
        else:
            raise ValueError(f"Unknown method: {self.method}")

        if valid_mask is not None:
            loss = loss * valid_mask.float()
            return loss.sum() / valid_mask.float().sum().clamp(min=1)

        return loss.mean()


class DecisionLoss(nn.Module):
    """决策损失：衡量动作选择的准确性。"""

    def __init__(self):
        super().__init__()

    def forward(self, choice_logits, target_actions, valid_mask=None):
        """计算决策损失。

        Args:
            choice_logits: (batch, num_actions) 选择 logits
            target_actions: (batch,) 真实最优动作
            valid_mask: (batch, num_actions) 有效动作 mask

        Returns:
            loss: 标量损失
        """
        if valid_mask is not None:
            masked_logits = choice_logits + (1 - valid_mask.float()) * (-1e9)
            loss = F.cross_entropy(masked_logits, target_actions)
        else:
            loss = F.cross_entropy(choice_logits, target_actions)

        return loss


class CombinedLoss(nn.Module):
    """联合损失：整合校准 + Q 值 + 决策损失。"""

    def __init__(self, config):
        super().__init__()
        self.calib_weight = config.get("calib_weight", 1.0)
        self.q_value_weight = config.get("q_value_weight", 1.0)
        self.decision_weight = config.get("decision_weight", 1.0)

        self.calib_loss_fn = CalibrationLoss(method="kl")
        self.q_value_loss_fn = QValueLoss(method="mse")
        self.decision_loss_fn = DecisionLoss()

    def forward(
        self,
        predicted_transitions,
        target_transitions,
        predicted_q_values,
        target_q_values,
        choice_logits,
        target_actions,
        valid_mask=None,
    ):
        calib_loss = self.calib_loss_fn(predicted_transitions, target_transitions, valid_mask)
        q_loss = self.q_value_loss_fn(predicted_q_values, target_q_values, valid_mask)
        dec_loss = self.decision_loss_fn(choice_logits, target_actions, valid_mask)

        total_loss = (
            self.calib_weight * calib_loss
            + self.q_value_weight * q_loss
            + self.decision_weight * dec_loss
        )

        return {
            "total_loss": total_loss,
            "calib_loss": calib_loss,
            "q_value_loss": q_loss,
            "decision_loss": dec_loss,
        }
