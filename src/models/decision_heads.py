import torch
import torch.nn as nn
import torch.nn.functional as F


class ChoiceHead(nn.Module):
    """动作选择打分器：对每个动作给出一个选择分数（动作数无关）。

    输入每个动作的特征向量 feat[a]，输出该动作的选择 logit；
    对所有动作的 logit 做 softmax 得到选择概率。
    """

    def __init__(self, feat_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, feat):
        """逐动作打分。

        Args:
            feat: (batch, num_actions, feat_dim) 每个动作的特征

        Returns:
            logits: (batch, num_actions) 每个动作的选择 logit
        """
        scores = self.net(feat)  # (batch, n, 1)
        return scores.squeeze(-1)

    def get_choice(self, feat):
        """获取确定的动作选择（argmax）与软概率。

        Args:
            feat: (batch, num_actions, feat_dim)

        Returns:
            choices: (batch,) 选择的动作索引
            probs: (batch, num_actions) 选择概率
        """
        logits = self.forward(feat)
        probs = F.softmax(logits, dim=-1)
        choices = torch.argmax(probs, dim=-1)
        return choices, probs


class TransitionHead(nn.Module):
    """转移概率打分器：对每个动作预测其到各结局的转移概率分布。

    输入每个动作的特征，输出该动作的 num_outcomes 维 logits，
    做 softmax 得到转移概率。
    """

    def __init__(self, feat_dim, hidden_dim, num_outcomes):
        super().__init__()
        self.num_outcomes = num_outcomes
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_outcomes),
        )

    def forward(self, feat):
        """逐动作预测转移概率。

        Args:
            feat: (batch, num_actions, feat_dim)

        Returns:
            transition_probs: (batch, num_actions, num_outcomes)
        """
        logits = self.net(feat)
        return F.softmax(logits, dim=-1)


class ReturnHead(nn.Module):
    """期望回报打分器：对每个动作预测其期望回报 Q(s,a)，动作数无关。"""

    def __init__(self, feat_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, feat):
        """逐动作预测期望回报。

        Args:
            feat: (batch, num_actions, feat_dim)

        Returns:
            expected_returns: (batch, num_actions)
        """
        return self.net(feat).squeeze(-1)


class ConfidenceHead(nn.Module):
    """置信度头：输出模型对当前决策的置信度（仅依赖场景向量）。"""

    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, scene_emb):
        """输出置信度 [0, 1]。

        Args:
            scene_emb: (batch, hidden_dim)

        Returns:
            confidence: (batch, 1) 置信度
        """
        return self.net(scene_emb)


class DecisionHeads(nn.Module):
    """决策头族：包含所有决策头的集合，逐动作打分，动作数可变。

    一次前向并行输出全部决策结果。
    """

    def __init__(self, config):
        super().__init__()
        hidden_dim = config.get("hidden_dim", 768)
        num_outcomes = config.get("num_outcomes", 2)

        # 领域 MoE 是否启用：启用时领域条件向量追加到逐动作特征末尾
        use_domain_experts = config.get("use_domain_experts", False)
        cond_dim = hidden_dim if use_domain_experts else 0

        # 逐动作特征维度 = [场景向量 ∥ 动作向量]，均为 hidden_dim
        feat_dim = hidden_dim + hidden_dim + cond_dim

        self.choice_head = ChoiceHead(feat_dim, hidden_dim)
        self.transition_head = TransitionHead(feat_dim, hidden_dim, num_outcomes)
        self.return_head = ReturnHead(feat_dim, hidden_dim)
        self.confidence_head = ConfidenceHead(hidden_dim)

    def forward(self, scene_emb, action_vec, num_valid_actions=None, cond=None):
        """并行前向传播，输出所有决策结果。

        Args:
            scene_emb: (batch, hidden_dim) 场景感知嵌入（attended_emb / embeddings）
            action_vec: (batch, num_actions, hidden_dim) 各动作特征向量
            num_valid_actions: (batch,) 每个样本的有效动作数（用于 mask 无效动作）
            cond: (batch, hidden_dim) 领域条件向量（领域 MoE 输出，可选）

        Returns:
            choices: (batch,)
            choice_probs: (batch, num_actions)
            transition_probs: (batch, num_actions, num_outcomes)
            expected_returns: (batch, num_actions)
            confidences: (batch, 1)
        """
        num_actions = action_vec.shape[1]
        scene_expanded = scene_emb.unsqueeze(1).expand(-1, num_actions, -1)
        feat = torch.cat([scene_expanded, action_vec], dim=-1)  # (batch, n, feat_dim)
        if cond is not None:
            cond_expanded = cond.unsqueeze(1).expand(-1, num_actions, -1)
            feat = torch.cat([feat, cond_expanded], dim=-1)

        choices, choice_probs = self.choice_head.get_choice(feat)
        transition_probs = self.transition_head(feat)
        expected_returns = self.return_head(feat)
        confidences = self.confidence_head(scene_emb)

        # 对超出有效动作数的 padding 位置做 mask，避免软选择吃到 padding
        if num_valid_actions is not None:
            mask = (
                torch.arange(num_actions, device=feat.device).unsqueeze(0)
                < num_valid_actions.unsqueeze(1)
            ).to(feat.dtype)
            logits = torch.log(choice_probs.clamp_min(1e-9))
            logits = logits * mask + (-1e9) * (1 - mask)
            choice_probs = torch.softmax(logits, dim=-1)
            choice_probs = choice_probs * mask
            choices = torch.argmax(choice_probs, dim=-1)

        return choices, choice_probs, transition_probs, expected_returns, confidences

    def get_transition_probs(self, scene_emb, action_vec, cond=None):
        """仅获取转移概率（用于校准评估）。"""
        feat = torch.cat(
            [
                scene_emb.unsqueeze(1).expand(-1, action_vec.shape[1], -1),
                action_vec,
            ],
            dim=-1,
        )
        if cond is not None:
            cond_expanded = cond.unsqueeze(1).expand(-1, action_vec.shape[1], -1)
            feat = torch.cat([feat, cond_expanded], dim=-1)
        return self.transition_head(feat)
