import torch
import pytest
from src.models.system_one_model import SystemOneModel
from src.sampler.parallel_sampler import ParallelSampler


class TestParallelSampler:
    def test_sampler_initialization(self):
        config = {
            "encoder_name": "distilbert-base-uncased",
            "encoder_freeze": True,
            "use_lora": False,
            "hf_offline": True,
            "hidden_dim": 128,
            "mdp_hidden_dim": 64,
            "num_actions": 3,
            "num_outcomes": 2,
            "num_bellman_steps": 4,
            "gamma": 0.9,
            "dropout": 0.1,
        }
        model = SystemOneModel(config)
        sampler = ParallelSampler(model)
        assert sampler.model is model

    def test_sample_output_shape(self):
        config = {
            "encoder_name": "distilbert-base-uncased",
            "encoder_freeze": True,
            "use_lora": False,
            "hf_offline": True,
            "hidden_dim": 128,
            "mdp_hidden_dim": 64,
            "num_actions": 3,
            "num_outcomes": 2,
            "num_bellman_steps": 4,
            "gamma": 0.9,
            "dropout": 0.1,
        }
        model = SystemOneModel(config)
        sampler = ParallelSampler(model)

        texts = ["Test input one", "Test input two"]
        result = sampler.sample(texts)

        assert "choices" in result
        assert "choice_probs" in result
        assert "transition_probs" in result
        assert "expected_returns" in result
        assert "confidences" in result

        assert result["choices"].shape == (2,)
        assert result["choice_probs"].shape == (2, 3)
        assert result["transition_probs"].shape == (2, 3, 2)
        assert result["expected_returns"].shape == (2, 3)
        assert result["confidences"].shape == (2,)

        assert 0 <= result["confidences"].max().item() <= 1
