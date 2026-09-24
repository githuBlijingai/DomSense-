import torch
import torch.nn.functional as F


class ReturnErrorEvaluator:
    """回报预测误差评估器。

    衡量模型的 expected_return 预测与真实回报之间的差距。
    """

    @staticmethod
    def _align(predicted, target):
        """对齐模型输出与数据目标的动作维度。

        模型固定输出 num_actions 维；数据按 batch 内有效动作数 padding。
        当两者维度不同时，将模型输出裁剪到数据维度。
        """
        if predicted.dim() == target.dim() and predicted.shape[-1] != target.shape[-1]:
            # 数据仅含有效动作数（2），模型输出完整动作数（3）
            predicted = predicted[..., : target.shape[-1]]
        return predicted, target

    @staticmethod
    def compute_mae(predicted, target, valid_mask=None):
        """平均绝对误差 (MAE)。"""
        predicted, target = ReturnErrorEvaluator._align(predicted, target)
        error = torch.abs(predicted - target)
        if valid_mask is not None:
            valid_mask = valid_mask[..., : target.shape[-1]]
            error = error * valid_mask.float()
            return error.sum() / valid_mask.float().sum().clamp(min=1)
        return error.mean()

    @staticmethod
    def compute_rmse(predicted, target, valid_mask=None):
        """均方根误差 (RMSE)。"""
        predicted, target = ReturnErrorEvaluator._align(predicted, target)
        error = (predicted - target) ** 2
        if valid_mask is not None:
            valid_mask = valid_mask[..., : target.shape[-1]]
            error = error * valid_mask.float()
            mse = error.sum() / valid_mask.float().sum().clamp(min=1)
        else:
            mse = error.mean()
        return torch.sqrt(mse)

    @staticmethod
    def compute_optimal_action_error(predicted_q, target_q):
        """最优动作选择误差：模型选的最优动作是否与真实最优一致。"""
        predicted_q, target_q = ReturnErrorEvaluator._align(predicted_q, target_q)
        predicted_best = torch.argmax(predicted_q, dim=1)
        target_best = torch.argmax(target_q, dim=1)
        return (predicted_best != target_best).float().mean()

    def evaluate(self, model, dataloader, device="cpu"):
        """全面评估回报预测误差。

        Args:
            model: SystemOneModel
            dataloader: DataLoader
            device: str

        Returns:
            results: dict
        """
        model.eval()
        all_mae = []
        all_rmse = []
        all_action_errors = []

        with torch.no_grad():
            for batch in dataloader:
                texts = batch["state_texts"]
                target_q = batch["expected_returns"].to(device)
                num_valid = batch.get("num_valid_actions")

                result = model(texts)
                predicted_q = result["expected_returns"]

                if num_valid is not None:
                    num_actions = target_q.shape[1]
                    valid_mask = torch.arange(num_actions, device=device).unsqueeze(0) < num_valid.unsqueeze(1)
                else:
                    valid_mask = None

                mae = self.compute_mae(predicted_q, target_q, valid_mask)
                rmse = self.compute_rmse(predicted_q, target_q, valid_mask)
                action_error = self.compute_optimal_action_error(predicted_q, target_q)

                all_mae.append(mae.item())
                all_rmse.append(rmse.item())
                all_action_errors.append(action_error.item())

        return {
            "mae": sum(all_mae) / len(all_mae),
            "rmse": sum(all_rmse) / len(all_rmse),
            "optimal_action_error_rate": sum(all_action_errors) / len(all_action_errors),
        }
