import torch
import pytest
from src.mdp import MDP, BellmanOperator
from src.mdp.reward import RewardFunction
from src.mdp.transition import TransitionEstimator


class TestMDP:
    def test_mdp_initialization(self):
        mdp = MDP(num_actions=3, num_outcomes=2, gamma=0.9)
        assert mdp.num_actions == 3
        assert mdp.num_outcomes == 2
        assert mdp.gamma == 0.9

    def test_expected_return(self):
        mdp = MDP(num_actions=3, num_outcomes=2, gamma=0.9)
        reward = torch.tensor([1.0])
        transition_probs = torch.tensor([[0.8, 0.2]])
        next_q = torch.tensor([[0.5, 0.3]])
        q = mdp.compute_expected_return(reward, transition_probs, next_q)
        expected = 1.0 + 0.9 * (0.8 * 0.5 + 0.2 * 0.3)
        assert torch.abs(q - expected) < 1e-6


class TestBellmanOperator:
    def test_bellman_iteration_convergence(self):
        operator = BellmanOperator(gamma=0.9, num_steps=10)
        batch_size = 4
        num_actions = 3
        num_outcomes = 2

        q_values = torch.rand(batch_size, num_actions, num_outcomes)
        rewards = torch.rand(batch_size, num_actions, 1)
        transition_probs = torch.softmax(torch.randn(batch_size, num_actions, num_outcomes), dim=-1)

        q_result, q_history = operator.iterate(q_values, rewards, transition_probs)

        assert q_result.shape == (batch_size, num_actions)
        assert len(q_history) == operator.num_steps + 1

        diffs = []
        for i in range(1, len(q_history)):
            diff = torch.max(torch.abs(q_history[i] - q_history[i - 1])).item()
            diffs.append(diff)

    def test_value_iteration(self):
        operator = BellmanOperator(gamma=0.9, num_steps=10)
        batch_size = 2
        num_actions = 2
        num_outcomes = 2

        q_values = torch.rand(batch_size, num_actions, num_outcomes)
        rewards = torch.rand(batch_size, num_actions, 1)
        transition_probs = torch.softmax(torch.randn(batch_size, num_actions, num_outcomes), dim=-1)

        q_fixed, n_iters = operator.value_iteration(q_values, rewards, transition_probs, tol=1e-4)
        q_step, _ = operator.iterate(q_values, rewards, transition_probs)

        assert q_fixed.shape == (batch_size, num_actions)
        assert n_iters > 0


class TestRewardFunction:
    def test_sparse_reward(self):
        rf = RewardFunction()
        selected = torch.tensor([1, 0, 2])
        target = torch.tensor([1, 2, 0])
        reward = rf.compute_sparse_reward(selected, target)
        assert reward[0] == 1.0
        assert reward[1] == 0.0
        assert reward[2] == 0.0


class TestTransitionEstimator:
    def test_kl_divergence(self):
        te = TransitionEstimator()
        predicted = torch.tensor([[[0.7, 0.3], [0.4, 0.6]]])
        target = torch.tensor([[[0.8, 0.2], [0.3, 0.7]]])
        kl = te.compute_kl_divergence(predicted, target)
        assert kl.shape == (1, 2)
        assert torch.all(kl >= 0)

    def test_brier_score(self):
        te = TransitionEstimator()
        predicted = torch.tensor([[[0.7, 0.3]]])
        target = torch.tensor([[[1.0, 0.0]]])
        brier = te.compute_brier_score(predicted, target)
        expected = (0.7 - 1.0) ** 2 + (0.3 - 0.0) ** 2
        assert torch.abs(brier[0, 0] - expected) < 1e-6

    def test_calibration(self):
        te = TransitionEstimator()
        predicted = torch.tensor([[[0.8, 0.2], [0.6, 0.4]]])
        target = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        ece, mce = te.compute_confidence_calibration(predicted, target)
        assert ece >= 0
        assert mce >= 0
