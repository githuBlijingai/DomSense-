import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardSoftmaxBaseline(nn.Module):
    """标准 Softmax 分类基线。

    不使用 MDP 前瞻，只是一个标准的分类头。
    用于对比实验，证明 MDP Head 的贡献。
    """

    def __init__(self, hidden_dim, num_actions):
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, num_actions)

    def forward(self, embeddings):
        return self.classifier(embeddings)


class NoMDPHeadModel(nn.Module):
    """无 MDP Head 的模型（消融基线）。

    只有编码器 + 决策头，没有隐式多步前瞻模块。
    """

    def __init__(self, config, encoder, decision_heads):
        super().__init__()
        self.encoder = encoder
        self.decision_heads = decision_heads

    def forward(self, texts):
        embeddings = self.encoder(texts)
        choices, choice_probs, transition_probs, expected_returns, confidences = (
            self.decision_heads(embeddings)
        )
        return {
            "choices": choices,
            "choice_probs": choice_probs,
            "transition_probs": transition_probs,
            "expected_returns": expected_returns,
            "confidences": confidences.squeeze(-1),
        }


class ComparativeEvaluator:
    """基线对比评估器。

    对比：
    1. 标准 Softmax 分类
    2. 无 MDP Head 模型
    3. 完整 System One 模型（含 MDP Head）
    """

    def __init__(self, device="cpu"):
        self.device = device

    def train_and_evaluate_baseline(self, model, train_loader, val_loader, num_epochs=5):
        """训练并评估基线模型。"""
        model.to(self.device)
        model.train()

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=5e-5,
        )

        for epoch in range(num_epochs):
            total_loss = 0.0
            for batch in train_loader:
                texts = batch["state_texts"]
                target_actions = batch["optimal_actions"].to(self.device)

                optimizer.zero_grad()
                result = model(texts)
                loss = F.cross_entropy(result["choice_probs"], target_actions)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                texts = batch["state_texts"]
                target_actions = batch["optimal_actions"].to(self.device)
                result = model(texts)
                predictions = torch.argmax(result["choice_probs"], dim=1)
                correct += (predictions == target_actions).sum().item()
                total += len(predictions)

        return correct / total if total > 0 else 0

    def compare_all(
        self,
        full_model,
        train_loader,
        val_loader,
        test_loader,
        encoder,
        decision_heads,
    ):
        """全面对比所有模型。

        Returns:
            results: dict 对比结果
        """
        from .calibration import CalibrationEvaluator

        calib_eval = CalibrationEvaluator()

        baseline_model = NoMDPHeadModel(
            full_model.config, encoder, decision_heads
        )

        results = {}

        full_model.eval()
        baseline_model.eval()

        full_calib = calib_eval.evaluate_model_calibration(
            full_model, test_loader, self.device
        )
        baseline_calib = calib_eval.evaluate_model_calibration(
            baseline_model, test_loader, self.device
        )

        results["full_model"] = full_calib
        results["no_mdp_head"] = baseline_calib

        results["improvement"] = {
            "ece_reduction": baseline_calib["ece"] - full_calib["ece"],
            "accuracy_improvement": full_calib["accuracy"] - baseline_calib["accuracy"],
        }

        return results
