import time
import torch


class BenchmarkEvaluator:
    """决策基准评测。

    评估维度：
    - 准确率 (Accuracy)
    - F1 分数
    - 校准误差 (ECE/MCE)
    - 延迟 (Latency)
    - 成本 (Cost / 参数量)
    """

    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device

    def measure_latency(self, texts, num_runs=10):
        """测量推理延迟。

        Args:
            texts: List[str]
            num_runs: int 重复次数

        Returns:
            latency_ms: float 平均延迟 (毫秒)
        """
        self.model.eval()
        self.model.to(self.device)

        if isinstance(texts, str):
            texts = [texts]

        with torch.no_grad():
            for _ in range(3):
                _ = self.model(texts)

        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = self.model(texts)
        end_time = time.time()

        avg_latency = (end_time - start_time) / num_runs * 1000
        return avg_latency

    def measure_throughput(self, texts, batch_size=32):
        """测量吞吐量。

        Args:
            texts: List[str]
            batch_size: int

        Returns:
            samples_per_sec: float
        """
        self.model.eval()
        total_samples = 0
        start_time = time.time()

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                _ = self.model(batch)
                total_samples += len(batch)

        end_time = time.time()
        elapsed = end_time - start_time
        return total_samples / elapsed if elapsed > 0 else 0

    def evaluate(self, model, dataloader, device="cpu"):
        """完整基准评测。"""
        from .calibration import CalibrationEvaluator
        from .return_error import ReturnErrorEvaluator

        model.eval()
        calib_eval = CalibrationEvaluator()
        return_eval = ReturnErrorEvaluator()
        total, correct = 0, 0

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in dataloader:
                texts = batch["state_texts"]
                target_actions = batch["optimal_actions"].to(device)

                result = model(texts)
                predictions = result["choices"]

                all_predictions.append(predictions.cpu())
                all_targets.append(target_actions.cpu())

                correct += (predictions == target_actions).sum().item()
                total += len(predictions)

        accuracy = correct / total if total > 0 else 0
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)

        return {
            "accuracy": accuracy,
            "n_samples": total,
        }
