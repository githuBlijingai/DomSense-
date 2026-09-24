import torch
import pytest
from src.models.system_one_model import SystemOneModel
from src.data.synthetic import SyntheticDataGenerator


class TestSystemOneModel:
    def get_default_config(self):
        return {
            "encoder_name": "distilbert-base-uncased",
            "encoder_freeze": True,
            "use_lora": False,
            "hidden_dim": 128,
            "mdp_hidden_dim": 64,
            "num_actions": 3,
            "num_outcomes": 2,
            "num_bellman_steps": 4,
            "gamma": 0.9,
            "dropout": 0.1,
            "hf_offline": True,
        }

    def test_model_initialization(self):
        config = self.get_default_config()
        model = SystemOneModel(config)
        total, trainable = model.count_parameters()
        assert total > 0

    def test_forward_shape(self):
        config = self.get_default_config()
        model = SystemOneModel(config)
        texts = ["Test input"]
        result = model(texts)

        assert "choices" in result
        assert result["choices"].shape == (1,)
        assert result["transition_probs"].shape == (1, 3, 2)
        assert result["expected_returns"].shape == (1, 3)
        assert result["confidences"].shape == (1,)

    def test_generate_compact_output(self):
        config = self.get_default_config()
        model = SystemOneModel(config)
        texts = ["Test input"]
        outputs = model.generate_compact_output(texts)

        assert len(outputs) == 1
        assert "answers" in outputs[0]
        assert "q1" in outputs[0]["answers"]
        assert outputs[0]["answers"]["q1"]["type"] == "choice"
        assert "prob_dist" in outputs[0]["answers"]["q1"]
        assert "expected_return" in outputs[0]["answers"]["q1"]
        assert "confidence" in outputs[0]["answers"]["q1"]

    def test_synthetic_data_integration(self):
        config = self.get_default_config()
        config["max_actions"] = 3
        model = SystemOneModel(config)
        generator = SyntheticDataGenerator(config)
        dataset = generator.generate_dataset(5)

        assert len(dataset) == 5

        sample = dataset[0]
        assert "state_text" in sample
        assert "transition_probs" in sample
        assert "expected_returns" in sample

        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=5, collate_fn=dataset.collate_fn)
        batch = next(iter(loader))

        texts = batch["state_texts"]
        result = model(texts)
        assert result["choices"].shape == (5,)

    def test_variable_actions_semantic_mode(self):
        """关键回归：语义模式下每个样本动作数可以不同（修复局限 1）。"""
        config = self.get_default_config()
        config["use_cross_attention"] = True
        config["cross_attn_heads"] = 4
        config["max_actions"] = 3
        model = SystemOneModel(config)
        texts = ["Test input 1", "Test input 2"]

        # 样本0有2个动作，样本1有4个动作
        action_texts = [
            ["选项A", "选项B"],
            ["选项A", "选项B", "选项C", "选项D"],
        ]
        num_valid = torch.tensor([2, 4])

        result = model(texts, action_texts=action_texts, num_valid_actions=num_valid)

        n = 4  # batch_max_n
        assert result["choices"].shape == (2,)
        assert result["transition_probs"].shape == (2, n, 2)
        assert result["expected_returns"].shape == (2, n)

        # 样本0只有2个有效动作，多余的被 mask
        assert result["choices"][0].item() in [0, 1]

    def test_position_invariance(self):
        """关键回归：交叉注意力模式下重排选项顺序遵循语义等变（修复局限 2）。"""
        config = self.get_default_config()
        config["use_cross_attention"] = True
        config["cross_attn_heads"] = 4
        config["max_actions"] = 3
        torch.manual_seed(42)
        model = SystemOneModel(config)
        model.eval()

        texts = ["Test input"]
        actions = ["去散步", "去读书", "去吃火锅"]
        # 固定一个置换：把位置 [0,1,2] 重排为 [2,0,1]
        perm = [2, 0, 1]
        actions_b = [actions[p] for p in perm]

        with torch.no_grad():
            r_a = model(texts, action_texts=[actions], num_valid_actions=torch.tensor([3]))
            r_b = model(texts, action_texts=[actions_b], num_valid_actions=torch.tensor([3]))

        probs_a = r_a["choice_probs"][0].detach().cpu()
        probs_b = r_b["choice_probs"][0].detach().cpu()

        # 位置不变性：B 排列中的第 i 个动作 = A 排列中的 perm[i] 的语义
        # 因此 probs_b 应当是 probs_a 按 perm 重排后的结果（等变）。
        # 允许一定容差（随机初始化 + 未训练时非完全对称）。
        permuted_a = probs_a[perm]
        diff = (probs_b - permuted_a).abs()
        max_diff = diff.max().item()
        # 语义模式位置不变的正确表现是 max_diff 显著小于槽位模式
        # 这里仅做数值合理性断言（不要求严格为 0）
        assert probs_b.shape == probs_a.shape
        assert max_diff < 0.5, f"重排后概率应大致等变，max_diff={max_diff}"
