from datasets import load_dataset
from .synthetic import MDPDataPoint, MDPDataset


class ExternalDataset:
    """公开数据集适配器。

    将公开分类/意图识别数据集转换为 MDP 决策链格式。
    由于公开数据通常没有真实的转移概率，我们使用模拟器生成近似 ground-truth。
    """

    @staticmethod
    def load_ag_news(split="train", max_samples=None):
        """加载 AG News 并转换为 MDP 格式。"""
        dataset = load_dataset("ag_news", split=split, trust_remote_code=True)

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        label_names = ["World", "Sports", "Business", "Science/Tech"]
        data_points = []

        for item in dataset:
            text = item["text"]
            true_label = item["label"]

            actions = label_names.copy()
            optimal_action = true_label

            num_outcomes = 2
            transition_probs = []
            for i in range(len(actions)):
                if i == optimal_action:
                    probs = [0.8, 0.2]
                else:
                    probs = [0.3, 0.7]
                transition_probs.append(probs)

            expected_returns = [0.9 if i == optimal_action else 0.2 for i in range(len(actions))]

            data_points.append(
                MDPDataPoint(
                    state_text=f"Classify the news: {text[:200]}",
                    actions=actions,
                    optimal_action=optimal_action,
                    transition_probs=transition_probs,
                    expected_returns=expected_returns,
                )
            )

        return MDPDataset(data_points)

    @staticmethod
    def load_trec(split="train", max_samples=None):
        """加载 TREC 问题分类数据。"""
        dataset = load_dataset("trec", split=split, trust_remote_code=True)

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        coarse_labels = ["DESC", "ENTY", "ABBR", "HUM", "NUM", "LOC"]
        data_points = []

        for item in dataset:
            text = item["text"]
            true_label = item["coarse_label"]

            actions = coarse_labels.copy()
            optimal_action = true_label

            transition_probs = []
            for i in range(len(actions)):
                if i == optimal_action:
                    probs = [0.85, 0.15]
                else:
                    probs = [0.25, 0.75]
                transition_probs.append(probs)

            expected_returns = [
                0.85 if i == optimal_action else 0.15 for i in range(len(actions))
            ]

            data_points.append(
                MDPDataPoint(
                    state_text=f"Question type classification: {text[:200]}",
                    actions=actions,
                    optimal_action=optimal_action,
                    transition_probs=transition_probs,
                    expected_returns=expected_returns,
                )
            )

        return MDPDataset(data_points)

    @staticmethod
    def load_dataset_by_name(name, split="train", max_samples=None):
        """按名称加载数据集。

        Args:
            name: str 数据集名称 ("ag_news" 或 "trec")
            split: str 数据分割
            max_samples: int 最大样本数

        Returns:
            MDPDataset
        """
        if name == "ag_news":
            return ExternalDataset.load_ag_news(split, max_samples)
        elif name == "trec":
            return ExternalDataset.load_trec(split, max_samples)
        else:
            raise ValueError(f"Unknown dataset: {name}")
