import torch
import torch.nn as nn


class ActionContextCrossAttention(nn.Module):
    """★ 交叉注意力：让场景向量"阅读"所有选项文本之后再进入 MDP 前瞻。

    把场景向量作为 query，所有选项编码作为 key/value，
    让模型感知"场景 + 每个选项的语义"关系，而非仅场景 + 位置槽位。

    Args:
        hidden_dim: 编码器输出维度
        num_heads: 注意力头数
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, state_emb, action_emb):
        """交叉注意力的前向传播。

        Args:
            state_emb: (batch, hidden_dim) 场景语义嵌入
            action_emb: (batch, num_actions, hidden_dim) 各选项语义嵌入

        Returns:
            attended: (batch, hidden_dim) 注意力后的场景感知向量
        """
        query = state_emb.unsqueeze(1)
        attended, _ = self.cross_attn(
            query=query, key=action_emb, value=action_emb
        )
        attended = attended.squeeze(1)
        attended = self.layer_norm(attended)
        return attended


class MDPHead(nn.Module):
    """★ 隐式多步前瞻 MDP Head（核心创新模块）。

    在潜空间执行 n 步 Bellman 迭代，模拟多步决策链前瞻。

    关键设计（逐动作架构）：
    - 每个动作由一个语义特征向量 action_vec[a] 表示（启用交叉注意力时来自
      选项文本编码；否则来自可学习的位置嵌入表）。
    - 所有 MLP 都是"逐动作打分器"：对每个动作独立计算，动作数量由运行时
      决定（由动作向量张量的 size 1 决定），不再依赖固定 num_actions 槽位。
    - 省略 one-hot 位置特征：启用交叉注意力时，动作差异完全由语义决定，
      模型无法借助"第几个槽位"作弊。
    - 跨动作信息通过"全局价值摘要"（对每个结局取所有动作的 max）注入
      Bellman 迭代，保留马尔可夫链"考虑其他动作的后续价值"语义。
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.get("hidden_dim", 768)
        self.mdp_hidden_dim = config.get("mdp_hidden_dim", 512)
        self.num_outcomes = config.get("num_outcomes", 2)
        self.num_bellman_steps = config.get("num_bellman_steps", 4)
        self.gamma = config.get("gamma", 0.9)
        self.dropout = nn.Dropout(config.get("dropout", 0.1))

        self.use_cross_attention = config.get("use_cross_attention", False)
        if self.use_cross_attention:
            self.cross_attn = ActionContextCrossAttention(
                hidden_dim=self.hidden_dim,
                num_heads=config.get("cross_attn_heads", 4),
            )

        # 槽位模式（未启用交叉注意力）时的可学习位置嵌入表。
        # 仅作为"第几个动作"的表示；启用交叉注意力时不会被使用。
        self.max_actions = config.get("max_actions", config.get("num_actions", 3))
        self.action_embed_table = nn.Embedding(self.max_actions, self.hidden_dim)

        # 领域 MoE 是否启用（由顶层 config.use_domain_experts 决定）。
        # 启用时，领域条件向量会追加到逐动作特征末尾，故 feat_dim 需 +hidden_dim。
        self.use_domain_experts = config.get("use_domain_experts", False)
        cond_dim = self.hidden_dim if self.use_domain_experts else 0

        # 逐动作特征维度 = [attended 场景向量 ∥ 该动作向量]，均为 hidden_dim
        feat_dim = self.hidden_dim + self.hidden_dim + cond_dim

        # q0：初始化每个动作各结局的价值
        self.q0_mlp = nn.Sequential(
            nn.Linear(feat_dim, self.mdp_hidden_dim),
            nn.LayerNorm(self.mdp_hidden_dim),
            nn.GELU(),
            self.dropout,
            nn.Linear(self.mdp_hidden_dim, self.num_outcomes),
        )

        # reward：预测每个动作的即时奖励
        self.reward_mlp = nn.Sequential(
            nn.Linear(feat_dim, self.mdp_hidden_dim),
            nn.GELU(),
            self.dropout,
            nn.Linear(self.mdp_hidden_dim, 1),
        )

        # bellman：用"逐动作特征 + 全局价值摘要"近似 Bellman 更新量
        val_dim = self.num_outcomes  # 全局价值摘要：每结局取所有动作的 max
        self.bellman_mlp = nn.Sequential(
            nn.Linear(feat_dim + val_dim, self.mdp_hidden_dim),
            nn.LayerNorm(self.mdp_hidden_dim),
            nn.GELU(),
            self.dropout,
            nn.Linear(self.mdp_hidden_dim, self.mdp_hidden_dim),
            nn.LayerNorm(self.mdp_hidden_dim),
            nn.GELU(),
            self.dropout,
            nn.Linear(self.mdp_hidden_dim, self.num_outcomes),
        )

    def forward(self, embeddings, action_vec=None, num_valid_actions=None, cond=None):
        """隐式多步 Bellman 前瞻（动作数可变）。

        Args:
            embeddings: (batch, hidden_dim) 场景语义嵌入
            action_vec: (batch, num_actions, hidden_dim) 每个动作的特征向量
                （语义模式 = 选项编码；槽位模式 = 位置嵌入）。None 时用位置嵌入。
            num_valid_actions: (batch,) 每个样本的有效动作数（用于 mask）
            cond: (batch, hidden_dim) 领域条件向量（领域 MoE 输出，可选）

        Returns:
            q_values: (batch, num_actions) 迭代后的 Q 值摘要
            q_history: list of (batch, num_actions) Q 值历史
            rewards: (batch, num_actions) 预测的奖励
            attended_emb: (batch, hidden_dim) 交叉注意力后的场景感知嵌入
        """
        batch_size = embeddings.shape[0]
        device = embeddings.device

        if self.use_cross_attention:
            attended_emb = self.cross_attn(embeddings, action_vec)
            attended_emb = attended_emb + embeddings
            act_vec = action_vec
        else:
            attended_emb = embeddings
            if action_vec is None:
                indices = torch.arange(self.max_actions, device=device).unsqueeze(
                    0
                ).repeat(batch_size, 1)
                act_vec = self.action_embed_table(indices)  # (batch, max_actions, hidden)
            else:
                act_vec = action_vec

        num_actions = act_vec.shape[1]

        # 逐动作特征：复制场景向量到每个动作，拼接该动作向量
        scene_expanded = attended_emb.unsqueeze(1).expand(-1, num_actions, -1)
        feat = torch.cat([scene_expanded, act_vec], dim=-1)  # (batch, n, feat_dim)

        # 领域条件：若启用领域 MoE，则把 cond 追加到每个动作的特征末尾
        if cond is not None:
            cond_expanded = cond.unsqueeze(1).expand(-1, num_actions, -1)
            feat = torch.cat([feat, cond_expanded], dim=-1)

        # q0 初始化（每个动作）
        q_values = self.q0_mlp(feat)  # (batch, n, num_outcomes)

        # 即时奖励（每个动作）
        rewards = self.reward_mlp(feat).squeeze(-1)  # (batch, n)

        # 隐式 Bellman 迭代
        q_history = []
        for _ in range(self.num_bellman_steps):
            # 全局价值摘要：对每个结局，取所有动作的最大价值（动作数无关）
            global_best, _ = q_values.max(dim=1)  # (batch, num_outcomes)
            global_best_expanded = global_best.unsqueeze(1).expand(-1, num_actions, -1)

            mdp_input = torch.cat([feat, global_best_expanded], dim=-1)
            delta = self.bellman_mlp(mdp_input)  # (batch, n, num_outcomes)
            q_values = q_values + delta

            q_summary, _ = q_values.max(dim=-1)  # (batch, n)
            q_history.append(q_summary.clone())

        if num_valid_actions is not None:
            mask = (
                torch.arange(num_actions, device=device).unsqueeze(0)
                < num_valid_actions.unsqueeze(1)
            )
            mask = mask.float()
            q_summary = q_summary * mask + (-1e9) * (1 - mask)

        return q_summary, q_history, rewards, attended_emb

    def get_expected_return(self, q_summary, transition_probs):
        """从 Q 值和转移概率计算期望回报。

        Args:
            q_summary: (batch, num_actions) Q 值
            transition_probs: (batch, num_actions, num_outcomes) 转移概率

        Returns:
            expected_return: (batch,) 最优动作的期望回报
        """
        best_actions = torch.argmax(q_summary, dim=1)
        best_q = q_summary[torch.arange(q_summary.shape[0]), best_actions]
        return best_q
