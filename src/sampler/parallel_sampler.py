import torch


class ParallelSampler:
    """并行采样器：一次前向输出所有决策结果。

    不需要自回归 token 生成，各决策头并行输出。
    """

    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device

    def sample(self, texts, return_mdp_details=False, num_valid_actions=None):
        """并行采样。

        Args:
            texts: List[str] 输入文本
            return_mdp_details: 是否返回 MDP 细节
            num_valid_actions: (batch,) 有效动作数

        Returns:
            result: dict
        """
        self.model.eval()
        with torch.no_grad():
            result = self.model(
                texts,
                return_mdp_details=return_mdp_details,
                num_valid_actions=num_valid_actions,
            )
        return result

    def batch_sample(self, texts, batch_size=32, return_mdp_details=False):
        """批量采样（支持大数据量分批处理）。

        Args:
            texts: List[str]
            batch_size: 每批大小
            return_mdp_details: 是否返回 MDP 细节

        Returns:
            accumulated: dict 累积结果
        """
        all_results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            result = self.sample(batch_texts, return_mdp_details)
            all_results.append(result)

        accumulated = {
            "choices": torch.cat([r["choices"] for r in all_results]),
            "choice_probs": torch.cat([r["choice_probs"] for r in all_results]),
            "transition_probs": torch.cat(
                [r["transition_probs"] for r in all_results]
            ),
            "expected_returns": torch.cat(
                [r["expected_returns"] for r in all_results]
            ),
            "q_values": torch.cat([r["q_values"] for r in all_results]),
            "confidences": torch.cat([r["confidences"] for r in all_results]),
        }

        if return_mdp_details and "q_history" in all_results[0]:
            accumulated["q_history"] = [
                torch.cat([r["q_history"][step] for r in all_results])
                for step in range(len(all_results[0]["q_history"]))
            ]
            accumulated["rewards"] = torch.cat(
                [r["rewards"] for r in all_results]
            )

        return accumulated

    def sample_with_different_temperatures(self, texts, temperatures=(0.5, 1.0, 2.0)):
        """在不同温度下采样（用于不确定性估计）。

        Args:
            texts: List[str]
            temperatures: Tuple[float] 温度值列表

        Returns:
            results: dict of temperature -> result
        """
        self.model.eval()
        results = {}

        with torch.no_grad():
            embeddings = self.model.encoder(texts).to(self.device)

            for temp in temperatures:
                logits = self.model.decision_heads.choice_head(embeddings)
                scaled_logits = logits / temp
                probs = torch.softmax(scaled_logits, dim=-1)
                choices = torch.multinomial(probs, num_samples=1).squeeze(-1)
                results[temp] = {
                    "choices": choices,
                    "probs": probs,
                }

        return results
