import torch


class CalibrationEvaluator:
    """校准指标评估器。

    计算 Expected Calibration Error (ECE) 和 Maximum Calibration Error (MCE)。
    这是本项目的核心评估指标 —— 概率校准被定义为状态转移概率估计。
    """

    def __init__(self, num_bins=10):
        self.num_bins = num_bins

    def compute_ece_mce(self, confidences, correct):
        """计算 ECE 和 MCE。

        Args:
            confidences: (N,) 预测置信度 (概率值)
            correct: (N,) 是否正确 (0/1)

        Returns:
            ece: float
            mce: float
        """
        device = confidences.device
        bin_boundaries = torch.linspace(0, 1, self.num_bins + 1, device=device)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        mce = 0.0

        for i in range(self.num_bins):
            in_bin = (confidences > bin_lowers[i]) & (confidences <= bin_uppers[i])
            prop_in_bin = in_bin.float().mean().item()

            if prop_in_bin > 0:
                avg_confidence = confidences[in_bin].mean().item()
                avg_accuracy = correct[in_bin].float().mean().item()
                bin_ece = abs(avg_confidence - avg_accuracy)
                ece += prop_in_bin * bin_ece
                mce = max(mce, bin_ece)

        return ece, mce

    def evaluate_transition_calibration(self, predicted_probs, target_probs):
        """评估转移概率预测的校准质量。

        Args:
            predicted_probs: (N, num_outcomes) 预测概率
            target_probs: (N, num_outcomes) 目标概率（通常为 one-hot）

        Returns:
            ece: float
            mce: float
        """
        batch_size = predicted_probs.shape[0]
        device = predicted_probs.device

        predicted_probs_flat = predicted_probs.reshape(-1, predicted_probs.shape[-1])
        target_probs_flat = target_probs.reshape(-1, target_probs.shape[-1])

        confidences, predictions = torch.max(predicted_probs_flat, dim=1)
        correct = (predictions == torch.argmax(target_probs_flat, dim=1)).float()

        return self.compute_ece_mce(confidences, correct)

    def evaluate_model_calibration(self, model, dataloader, device="cpu"):
        """评估完整模型的校准性能。

        Args:
            model: SystemOneModel
            dataloader: DataLoader
            device: str

        Returns:
            results: dict
        """
        model.eval()
        all_confidences = []
        all_correct = []

        with torch.no_grad():
            for batch in dataloader:
                texts = batch["state_texts"]
                target_actions = batch["optimal_actions"].to(device)

                result = model(texts)
                choice_probs = result["choice_probs"]
                confidences, predictions = torch.max(choice_probs, dim=1)
                correct = (predictions == target_actions).float()

                all_confidences.append(confidences.cpu())
                all_correct.append(correct.cpu())

        all_confidences = torch.cat(all_confidences)
        all_correct = torch.cat(all_correct)

        ece, mce = self.compute_ece_mce(all_confidences, all_correct)

        accuracy = all_correct.float().mean().item()

        return {
            "ece": ece,
            "mce": mce,
            "accuracy": accuracy,
            "n_samples": len(all_confidences),
        }

    def reliability_diagram_data(self, confidences, correct):
        """生成可靠性图数据。

        Returns:
            bin_data: list of dict
        """
        device = confidences.device
        bin_boundaries = torch.linspace(0, 1, self.num_bins + 1, device=device)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        bin_data = []
        for i in range(self.num_bins):
            in_bin = (confidences > bin_lowers[i]) & (confidences <= bin_uppers[i])
            count = in_bin.sum().item()
            if count > 0:
                avg_conf = confidences[in_bin].mean().item()
                avg_acc = correct[in_bin].float().mean().item()
            else:
                avg_conf = (bin_lowers[i] + bin_uppers[i]).item() / 2
                avg_acc = 0.0

            bin_data.append(
                {
                    "bin": i,
                    "bin_lower": bin_lowers[i].item(),
                    "bin_upper": bin_uppers[i].item(),
                    "count": count,
                    "avg_confidence": avg_conf,
                    "avg_accuracy": avg_acc,
                }
            )

        return bin_data
