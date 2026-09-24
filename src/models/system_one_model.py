import torch
import torch.nn as nn

from .encoder import LLMEncoder
from .mdp_head import MDPHead
from .decision_heads import DecisionHeads
from .domain_moe import DomainMoE


class SystemOneModel(nn.Module):
    """System One 决策链模型：完整组装。

    架构：
    1. LLM 编码器: 文本 → 语义嵌入（潜空间）
    2. MDP Head: 隐式多步 Bellman 前瞻
    3. 决策头族: 并行输出 Choice / Transition / Return / Confidence

    关键设计：内部隐式多步前瞻，对外紧凑下一步输出。

    +-- 2025-09-24: 逐动作架构重构 --+
    所有 MLP 接受 [scene_emb ∥ action_vec[a]] 作为特征，对每个动作独立打分，
    动作数 n 由运行时 action_vec 的形状决定，
    不再依赖固定 num_actions 槽位，不再使用 one-hot 位置编码。
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_outcomes = config.get("num_outcomes", 2)
        self.use_cross_attention = config.get("use_cross_attention", False)

        self.encoder = LLMEncoder(config)
        self.mdp_head = MDPHead(config)
        self.decision_heads = DecisionHeads(config)

        # 领域专家 MoE（默认禁用：config.domains 为空时自动关闭）
        self.use_domain_experts = config.get("use_domain_experts", False)
        self.domain_moe = DomainMoE(config)

    def encode_options(self, actions_texts, device=None):
        """编码选项文本列表为动作语义嵌入（支持每样本不同动作数）。

        Args:
            actions_texts: list of list[str]，每个样本的选项文本列表
                （每样本动作数可以不同）
            device: 目标设备

        Returns:
            action_embeddings: (batch, batch_max_n, hidden_dim) 每个选项的语义嵌入
            num_valid: (batch,) 每个样本的有效动作数
        """
        batch_size = len(actions_texts)
        num_valid = torch.tensor(
            [len(row) for row in actions_texts], dtype=torch.long
        )
        batch_max_n = num_valid.max().item()

        flat_texts = []
        for row in actions_texts:
            flat_texts.extend(row)
            # padding 到 batch_max_n；padding 条目用占位符，编码后会被 mask
            pad_count = batch_max_n - len(row)
            flat_texts.extend(["无" for _ in range(pad_count)])

        flat_emb = self.encoder(flat_texts)
        action_embeddings = flat_emb.view(batch_size, batch_max_n, -1)
        if device is not None:
            action_embeddings = action_embeddings.to(device)
            num_valid = num_valid.to(device)
        return action_embeddings, num_valid

    def _build_action_vec(self, embeddings, action_texts, num_valid_actions):
        """构建 action_vec: (batch, n, hidden_dim)。

        语义模式：对各选项文本编码。
        槽位模式：用可学习的位置嵌入表。
        """
        if action_texts is not None:
            action_embeddings, inferred_num_valid = self.encode_options(
                action_texts, device=embeddings.device
            )
            if num_valid_actions is None:
                num_valid_actions = inferred_num_valid
            return action_embeddings, num_valid_actions
        else:
            max_n = self.mdp_head.max_actions
            n = num_valid_actions.max().item() if num_valid_actions is not None else max_n
            indices = (
                torch.arange(max_n, device=embeddings.device)
                .unsqueeze(0)
                .expand(embeddings.shape[0], -1)
            )
            act_vec = self.mdp_head.action_embed_table(indices)
            if n < max_n:
                act_vec = act_vec[:, :n, :]
            return act_vec, num_valid_actions

    def _compute_domain_cond(self, scene_emb):
        """计算领域条件向量（领域 MoE 自动路由）。

        无需外部传入领域标签——门控网络从场景语义自动判断激活哪些专家。

        Args:
            scene_emb: (batch, hidden_dim) 场景语义嵌入

        Returns:
            cond: (batch, hidden_dim) 领域条件向量；未启用时返回 None
        """
        if not self.use_domain_experts or not self.domain_moe.enabled:
            return None
        return self.domain_moe(scene_emb)

    def forward(
        self,
        texts,
        action_texts=None,
        return_mdp_details=False,
        num_valid_actions=None,
    ):
        """完整前向传播。

        Args:
            texts: List[str] 输入文本
            action_texts: list of list[str]，每个样本的选项文本（可选）
            return_mdp_details: 是否返回 MDP 内部细节
            num_valid_actions: (batch,) 有效动作数（用于 padding mask）

        Returns:
            result: dict 包含所有决策输出
        """
        embeddings = self.encoder(texts)

        action_vec, num_valid = self._build_action_vec(
            embeddings, action_texts, num_valid_actions
        )

        # 领域条件向量由门控网络从场景语义自动路由，无需外部领域标签
        cond = self._compute_domain_cond(embeddings)

        q_summary, q_history, rewards, attended_emb = self.mdp_head(
            embeddings, action_vec, num_valid, cond
        )

        choices, choice_probs, transition_probs, expected_returns, confidences = (
            self.decision_heads(attended_emb, action_vec, num_valid, cond)
        )

        result = {
            "choices": choices,
            "choice_probs": choice_probs,
            "transition_probs": transition_probs,
            "expected_returns": expected_returns,
            "q_values": q_summary,
            "confidences": confidences.squeeze(-1),
            "embeddings": embeddings,
        }

        if return_mdp_details:
            result["q_history"] = q_history
            result["rewards"] = rewards

        return result

    def generate_compact_output(self, texts, action_texts=None, num_valid_actions=None):
        """生成紧凑的下一步输出（符合输出契约）。

        Args:
            texts: List[str] 输入文本
            action_texts: list of list[str]，每个样本的选项文本（可选）
            num_valid_actions: (batch,) 有效动作数

        Returns:
            outputs: List[dict] 每个样本的紧凑输出
        """
        embeddings = self.encoder(texts)

        action_vec, num_valid = self._build_action_vec(
            embeddings, action_texts, num_valid_actions
        )

        # 领域条件向量由门控网络从场景语义自动路由
        cond = self._compute_domain_cond(embeddings)

        q_summary, q_history, rewards, attended_emb = self.mdp_head(
            embeddings, action_vec, num_valid, cond
        )

        choices, choice_probs, transition_probs, expected_returns, confidences = (
            self.decision_heads(attended_emb, action_vec, num_valid, cond)
        )

        n_actions = action_vec.shape[1]
        outputs = []
        batch_size = len(texts)
        for i in range(batch_size):
            transition_dict = {}
            for a in range(n_actions):
                transition_dict[f"action_{a}"] = (
                    transition_probs[i, a].detach().cpu().tolist()
                )

            output = {
                "answers": {
                    "q1": {
                        "type": "choice",
                        "choice": f"选项{choices[i].item() + 1}",
                        "prob_dist": transition_dict,
                        "expected_return": round(
                            expected_returns[i, choices[i]].item(), 4
                        ),
                        "confidence": round(confidences[i].item(), 4),
                    }
                }
            }
            outputs.append(output)

        return outputs

    def get_trainable_parameters(self):
        """获取可训练参数（用于优化器）。"""
        trainable_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        return trainable_params

    def count_parameters(self):
        """统计参数数量。"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
