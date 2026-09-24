import torch
import pytest
from src.models.mdp_head import MDPHead


class TestMDPHead:
    def test_initialization(self):
        config = {
            "hidden_dim": 128,
            "mdp_hidden_dim": 64,
            "num_actions": 3,
            "num_outcomes": 2,
            "num_bellman_steps": 4,
            "gamma": 0.9,
            "dropout": 0.1,
        }
        head = MDPHead(config)
        assert head.num_bellman_steps == 4
        assert head.hidden_dim == 128
        assert head.num_outcomes == 2
        assert head.max_actions == 3

    def test_forward_slot_mode(self):
        """槽位模式（无交叉注意力）：不传 action_vec 用位置嵌入表。"""
        config = {
            "hidden_dim": 128,
            "mdp_hidden_dim": 64,
            "num_actions": 3,
            "max_actions": 3,
            "num_outcomes": 2,
            "num_bellman_steps": 4,
            "gamma": 0.9,
            "dropout": 0.1,
        }
        head = MDPHead(config)
        batch_size = 4
        embeddings = torch.randn(batch_size, 128)

        q_summary, q_history, rewards, attended_emb = head(embeddings)

        assert q_summary.shape == (batch_size, 3)
        assert len(q_history) == 4
        assert rewards.shape == (batch_size, 3)
        assert attended_emb.shape == (batch_size, 128)
        # 槽位模式下 attended_emb 就是 embeddings 本身
        assert torch.equal(attended_emb, embeddings)

    def test_forward_variable_actions(self):
        """关键回归：动作数可变，4 个动作也正确工作。"""
        config = {
            "hidden_dim": 128,
            "mdp_hidden_dim": 64,
            "num_actions": 3,
            "max_actions": 6,
            "num_outcomes": 2,
            "num_bellman_steps": 4,
            "gamma": 0.9,
            "dropout": 0.1,
        }
        head = MDPHead(config)
        batch_size = 4
        embeddings = torch.randn(batch_size, 128)

        # 语义模式：action_vec (batch, n, hidden)，n=5
        n = 5
        action_vec = torch.randn(batch_size, n, 128)

        q_summary, q_history, rewards, attended_emb = head(
            embeddings, action_vec
        )

        assert q_summary.shape == (batch_size, n)
        assert rewards.shape == (batch_size, n)
        assert attended_emb.shape == (batch_size, 128)

    def test_forward_with_valid_mask(self):
        """不同样本有效动作数不同时，q_summary 被 mask。"""
        config = {
            "hidden_dim": 128,
            "mdp_hidden_dim": 64,
            "num_actions": 3,
            "max_actions": 3,
            "num_outcomes": 2,
            "num_bellman_steps": 4,
            "gamma": 0.9,
            "dropout": 0.1,
        }
        head = MDPHead(config)
        batch_size = 4
        embeddings = torch.randn(batch_size, 128)
        num_valid = torch.tensor([2, 3, 1, 2])

        q_summary, q_history, rewards, attended_emb = head(
            embeddings, num_valid_actions=num_valid
        )

        assert q_summary.shape == (batch_size, 3)
        # mask 后，超过有效动作数的位置应被压到极小值
        assert q_summary[0, 2].item() < -1e8
        assert q_summary[2, 1].item() < -1e8
        assert q_summary[2, 2].item() < -1e8

    def test_cross_attention_mode(self):
        """交叉注意力模式：action_vec 必须提供且 shape 正确。"""
        config = {
            "hidden_dim": 128,
            "mdp_hidden_dim": 64,
            "num_actions": 3,
            "max_actions": 3,
            "num_outcomes": 2,
            "num_bellman_steps": 4,
            "gamma": 0.9,
            "dropout": 0.1,
            "use_cross_attention": True,
            "cross_attn_heads": 4,
        }
        head = MDPHead(config)
        batch_size = 4
        n = 4
        embeddings = torch.randn(batch_size, 128)
        action_vec = torch.randn(batch_size, n, 128)

        q_summary, q_history, rewards, attended_emb = head(
            embeddings, action_vec
        )

        assert q_summary.shape == (batch_size, n)
        assert rewards.shape == (batch_size, n)
        assert len(q_history) == 4
        assert attended_emb.shape == (batch_size, 128)
        # 交叉注意力模式下 attended_emb 与 embeddings 不同（经过注意力后）
        assert not torch.equal(attended_emb, embeddings)

    def test_bellman_steps(self):
        config = {
            "hidden_dim": 128,
            "mdp_hidden_dim": 64,
            "num_actions": 2,
            "max_actions": 2,
            "num_outcomes": 2,
            "num_bellman_steps": 5,
            "gamma": 0.9,
            "dropout": 0.0,
        }
        head = MDPHead(config)
        embeddings = torch.randn(4, 128)

        q1, h1, r1, a1 = head(embeddings)

        head.num_bellman_steps = 10
        q2, h2, r2, a2 = head(embeddings)

        assert len(h1) == 5
        assert len(h2) == 10
