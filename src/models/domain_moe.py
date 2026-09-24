"""领域专家路由（Domain MoE）。

在不改动共享编码器（encoder / cross-attention）的前提下，为不同领域提供
"领域条件化"的决策打分能力。设计目标：

- 领域路由是轻量的：只对下游打分头做条件化，不复制巨大的 backbone。
- 领域共享底面：encoder、cross-attention、q0/reward/bellman 主网络仍是全局共享，
  保留跨领域通用知识（软迁移的关键：共享权重可从旧 checkpoint 直接加载）。
- 专家差异：每个领域有一个轻量 Adapter 学习该领域的"可迁移偏移"，
  通过门控权重混合进逐动作特征，缓解跨领域训练时的相互干扰。

结构：

  DomainMoE
      domain_table  : Embedding(num_domains, hidden_dim)   每个领域的条件向量
      gate          : Linear(hidden_dim, num_domains)       门控（从场景注意力向量出）
      experts       : ModuleList[Adapter] × num_domains     每领域一个轻量专家
      num_domains   : 由 config 中 domains 列表长度决定
      特殊槽位       : "UNK" 用于未登记领域（回退到通用专家）

默认关闭：仅当 config.use_domain_experts 为 True 时才启用，且默认 domains=[]，
保证旧 checkpoint 与现有全部测试在未开启时行为完全不变。
"""

import torch
import torch.nn as nn

UNK_DOMAIN = "__UNK__"


class DomainMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.get("hidden_dim", 768)
        domains = config.get("domains", [])
        self.domains = list(domains)

        # 归一化领域名单
        ids = []
        for d in self.domains:
            d = str(d).strip()
            if d and d != UNK_DOMAIN:
                ids.append(d)
        self.domains = ids
        self.index = {d: i for i, d in enumerate(self.domains)}

        num_domains = len(self.domains)
        num_slots = num_domains + 1  # 最后一个槽位留给 UNK / 兜底

        if num_domains == 0:
            self.enabled = False
            self.register_buffer("slot_count", torch.tensor(0))
        else:
            self.enabled = True
            self.register_buffer("slot_count", torch.tensor(num_slots))

            expert_hidden = config.get("domain_expert_hidden", self.hidden_dim)
            gate_hidden = config.get("domain_gate_hidden", min(self.hidden_dim, 256))

            self.domain_table = nn.Embedding(num_slots, self.hidden_dim)
            # 门控：把"全局 / 目标领域的语义"转成对 expert 的加权
            self.gate = nn.Sequential(
                nn.Linear(self.hidden_dim, gate_hidden),
                nn.GELU(),
                nn.Linear(gate_hidden, num_slots),
            )
            # 每领域一个轻量 Adapter（残差，"领域偏移"）
            experts = []
            for _ in range(num_slots):
                experts.append(
                    nn.Sequential(
                        nn.Linear(self.hidden_dim, expert_hidden),
                        nn.GELU(),
                        nn.Dropout(config.get("dropout", 0.1)),
                        nn.Linear(expert_hidden, self.hidden_dim),
                    )
                )
            self.experts = nn.ModuleList(experts)

    def slot_id(self, domain: str | None):
        if not domain:
            return len(self.domains)  # UNK 槽位
        domain = str(domain).strip()
        return self.index.get(domain, len(self.domains))

    def forward(self, scene_emb):
        """根据场景语义自动路由领域专家，产出领域条件向量。

        不再依赖外部传入的 domain 标签——门控网络从场景向量自动学习
        该激活哪些专家，领域向量也由门控权重对嵌入表加权得到，
        实现"题目内容 → 自动领域分类 → 专家路由"的端到端自动决策。

        Args:
            scene_emb: (batch, hidden_dim) 场景语义嵌入

        Returns:
            cond: (batch, hidden_dim) 领域条件向量，追加到逐动作特征中
        """
        if not self.enabled:
            return None
        device = scene_emb.device

        # 门控权重：从场景语义自动判断各领域专家的激活程度
        gate_logits = self.gate(scene_emb)  # (batch, slots)
        gate_weights = torch.softmax(gate_logits, dim=-1)  # (batch, slots)

        # 领域向量：用门控权重对领域嵌入表加权求和（不再查表，完全自动）
        dom_emb = (gate_weights.unsqueeze(-1) * self.domain_table.weight).sum(
            dim=1
        )  # (batch, hidden)

        # 专家加权聚合（MoE 聚合）：每个样本用 gate 加权所有 expert 的输出
        expert_outs = torch.stack(
            [e(scene_emb) for e in self.experts], dim=1
        )  # (batch, slots, hidden)
        mixed = (gate_weights.unsqueeze(-1) * expert_outs).sum(
            dim=1
        )  # (batch, hidden)

        return dom_emb + mixed
