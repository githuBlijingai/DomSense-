import torch
import pytest
from src.eval.calibration import CalibrationEvaluator
from src.eval.return_error import ReturnErrorEvaluator
from src.training.losses import CalibrationLoss, QValueLoss, DecisionLoss


class TestCalibrationEvaluator:
    def test_perfect_calibration(self):
        evaluator = CalibrationEvaluator(num_bins=10)
        confidences = torch.tensor([0.9, 0.8, 0.7, 0.6])
        correct = torch.tensor([1, 1, 1, 1])
        ece, mce = evaluator.compute_ece_mce(confidences, correct)
        assert ece >= 0
        assert mce >= 0

    def test_ece_zero_for_perfect(self):
        evaluator = CalibrationEvaluator(num_bins=10)
        confidences = torch.full((100,), 0.5)
        correct = torch.ones(100)
        for i in range(50):
            correct[i] = 0
        ece, mce = evaluator.compute_ece_mce(confidences, correct)
        assert abs(ece - 0.0) < 0.01


class TestReturnErrorEvaluator:
    def test_perfect_prediction(self):
        evaluator = ReturnErrorEvaluator()
        predicted = torch.tensor([[1.0, 0.5, 0.0]])
        target = torch.tensor([[1.0, 0.5, 0.0]])
        mae = evaluator.compute_mae(predicted, target)
        assert mae.item() == 0.0

    def test_optimal_action_correct(self):
        evaluator = ReturnErrorEvaluator()
        predicted = torch.tensor([[0.9, 0.1, 0.5]])
        target = torch.tensor([[0.8, 0.2, 0.3]])
        error = evaluator.compute_optimal_action_error(predicted, target)
        assert error.item() == 0.0

    def test_mae_dim_mismatch_alignment(self):
        """回归：模型输出 3 个动作，数据仅含 2 个有效动作时维度不匹配。"""
        evaluator = ReturnErrorEvaluator()
        predicted = torch.tensor([[1.0, 0.5, 2.0]])  # (1, 3)
        target = torch.tensor([[1.0, 0.5]])  # (1, 2)
        mae = evaluator.compute_mae(predicted, target)
        assert mae.item() == 0.0

    def test_rmse_dim_mismatch_alignment(self):
        """回归：RMSE 下模型输出与目标动作维度不一致。"""
        evaluator = ReturnErrorEvaluator()
        predicted = torch.tensor([[1.0, 0.5, 2.0]])  # (1, 3)
        target = torch.tensor([[1.0, 0.5]])  # (1, 2)
        rmse = evaluator.compute_rmse(predicted, target)
        assert rmse.item() == 0.0


class TestLosses:
    def test_calibration_loss_kl(self):
        loss_fn = CalibrationLoss(method="kl")
        pred = torch.tensor([[[0.7, 0.3], [0.4, 0.6]]])
        target = torch.tensor([[[0.8, 0.2], [0.3, 0.7]]])
        loss = loss_fn(pred, target)
        assert loss.item() >= 0

    def test_calibration_loss_brier(self):
        loss_fn = CalibrationLoss(method="brier")
        pred = torch.tensor([[[1.0, 0.0]]])
        target = torch.tensor([[[1.0, 0.0]]])
        loss = loss_fn(pred, target)
        assert loss.item() == 0.0

    def test_q_value_loss_mse(self):
        loss_fn = QValueLoss(method="mse")
        pred = torch.tensor([[1.0, 2.0, 3.0]])
        target = torch.tensor([[1.0, 2.0, 3.0]])
        loss = loss_fn(pred, target)
        assert loss.item() == 0.0

    def test_decision_loss(self):
        loss_fn = DecisionLoss()
        logits = torch.tensor([[2.0, 1.0, 0.1]])
        targets = torch.tensor([0])
        loss = loss_fn(logits, targets)
        assert loss.item() > 0
