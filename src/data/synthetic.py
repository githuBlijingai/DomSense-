import random
import torch
from torch.utils.data import Dataset


class MDPDataPoint:
    """单个 MDP 决策数据点。"""

    def __init__(
        self,
        state_text,
        actions,
        optimal_action,
        transition_probs,
        expected_returns,
        num_outcomes=2,
        domain=None,
    ):
        self.state_text = state_text
        self.actions = actions
        self.optimal_action = optimal_action
        self.transition_probs = transition_probs
        self.expected_returns = expected_returns
        self.num_outcomes = num_outcomes
        self.domain = domain or "general"


class MDPDataset(Dataset):
    """MDP 决策链数据集。"""

    def __init__(self, data_points):
        self.data_points = data_points

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, idx):
        dp = self.data_points[idx]

        transition_probs = torch.tensor(dp.transition_probs, dtype=torch.float32)
        expected_returns = torch.tensor(dp.expected_returns, dtype=torch.float32)

        return {
            "state_text": dp.state_text,
            "actions": dp.actions,
            "optimal_action": torch.tensor(dp.optimal_action, dtype=torch.long),
            "transition_probs": transition_probs,
            "expected_returns": expected_returns,
            "num_actions": len(dp.actions),
            "domain": dp.domain,
        }

    @staticmethod
    def collate_fn(batch):
        texts = [item["state_text"] for item in batch]
        optimal_actions = torch.stack([item["optimal_action"] for item in batch])
        num_actions_list = [item["num_actions"] for item in batch]
        domains = [item.get("domain") or "general" for item in batch]

        max_actions = max(num_actions_list)
        num_outcomes = batch[0]["transition_probs"].shape[-1]

        batch_transition_probs = []
        batch_expected_returns = []
        batch_action_texts = []

        for item in batch:
            tp = item["transition_probs"]
            er = item["expected_returns"]
            acts = list(item["actions"])
            if tp.shape[0] < max_actions:
                pad_size = max_actions - tp.shape[0]
                tp = torch.cat([tp, torch.zeros(pad_size, num_outcomes)], dim=0)
                er = torch.cat([er, torch.zeros(pad_size)], dim=0)
                acts = acts + ["无" for _ in range(pad_size)]
            batch_transition_probs.append(tp)
            batch_expected_returns.append(er)
            batch_action_texts.append(acts)

        return {
            "state_texts": texts,
            "action_texts": batch_action_texts,
            "optimal_actions": optimal_actions,
            "transition_probs": torch.stack(batch_transition_probs),
            "expected_returns": torch.stack(batch_expected_returns),
            "num_valid_actions": torch.tensor(num_actions_list),
            "domains": domains,
        }


class SyntheticDataGenerator:
    """自造 MDP 决策链数据生成器。

    生成带明确真实转移概率的决策数据，可用于验证校准。
    """

    DECISION_TEMPLATES = [
        "用户需要选择最佳的方案来处理{context}。当前选项：{options}",
        "基于以下情况{context}，应该选择哪个方案？选项：{options}",
        "分析{context}，从以下方案中选择最优的一个：{options}",
    ]

    CONTEXTS = [
        "一个技术团队的技术选型问题",
        "一家公司面对的市场竞争策略",
        "一个学生选择学习路线的决策",
        "一个投资者配置资产组合",
        "一个项目经理的资源分配问题",
        "一个用户选择软件工具的过程",
        "一个团队面对的架构设计决策",
        "一个企业数字化转型方案",
        "一个研究人员选择实验方法",
        "一个创业者选择产品的方向",
    ]

    ACTION_TEMPLATES = [
        ["方案A：保守策略", "方案B：激进策略", "方案C：均衡策略"],
        ["采用开源方案", "购买商业方案", "自研定制方案"],
        ["优先提升质量", "优先降低成本", "优先加快速度"],
        ["集中资源于单一方向", "分散资源多元布局", "优先保障核心业务"],
        ["立即执行方案", "分阶段逐步推进", "先验证再推广"],
    ]

    def __init__(self, config):
        self.num_outcomes = config.get("num_outcomes", 2)
        self.max_actions = config.get("max_actions", 3)

    def generate_transition_probs(self, num_actions):
        """生成真实的转移概率矩阵（可用作校准 ground-truth）。"""
        probs = []
        for _ in range(num_actions):
            logits = [random.uniform(0, 1) for _ in range(self.num_outcomes)]
            total = sum(logits)
            probs.append([p / total for p in logits])
        return probs

    def generate_expected_returns(self, transition_probs):
        """根据转移概率计算期望回报。"""
        returns = []
        for tp in transition_probs:
            outcomes = [random.uniform(0, 10) for _ in range(len(tp))]
            er = sum(p * o for p, o in zip(tp, outcomes))
            returns.append(round(er, 2))
        return returns

    def generate_datapoint(self):
        """生成一个 MDP 数据点。"""
        context = random.choice(self.CONTEXTS)
        action_set = random.choice(self.ACTION_TEMPLATES)

        num_actions = random.randint(2, min(3, len(action_set)))
        selected_actions = action_set[:num_actions]

        transition_probs = self.generate_transition_probs(num_actions)
        expected_returns = self.generate_expected_returns(transition_probs)

        optimal_action = int(torch.argmax(torch.tensor(expected_returns)).item())

        template = random.choice(self.DECISION_TEMPLATES)
        options_str = "; ".join(
            [f"{i+1}. {a}" for i, a in enumerate(selected_actions)]
        )
        state_text = template.format(context=context, options=options_str)

        return MDPDataPoint(
            state_text=state_text,
            actions=selected_actions,
            optimal_action=optimal_action,
            transition_probs=transition_probs,
            expected_returns=expected_returns,
            num_outcomes=self.num_outcomes,
        )

    def generate_dataset(self, num_samples):
        """生成数据集。

        Args:
            num_samples: int 样本数量

        Returns:
            MDPDataset
        """
        data_points = [self.generate_datapoint() for _ in range(num_samples)]
        return MDPDataset(data_points)
