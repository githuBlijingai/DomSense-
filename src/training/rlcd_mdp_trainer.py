import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

from .losses import CombinedLoss
from src.utils.log import Logger


class RLCDMDPTrainer:
    """★ RLCD-MDP 训练器。

    联合训练循环：
    1. 编码器 + MDP 前瞻前向
    2. 各头并行输出
    3. 计算校准 + Q + 决策损失
    4. 反向传播更新（含 LoRA）
    """

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device("cpu")

        model.to(self.device)

        trainable_params = model.get_trainable_parameters()
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=float(config.get("learning_rate", 5e-5)),
            weight_decay=float(config.get("weight_decay", 0.01)),
        )

        total_steps = config.get("max_steps", 1000)
        warmup_steps = config.get("warmup_steps", 100)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps - warmup_steps
        )

        self.loss_fn = CombinedLoss(config)
        self.logger = Logger.get()

        self.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 2)
        self.max_steps = config.get("max_steps", 1000)
        self.eval_every = config.get("eval_every", 50)
        self.save_every = config.get("save_every", 100)
        self.output_dir = config.get("output_dir", "./outputs")

        self.best_val_loss = float("inf")
        self.global_step = 0

    def train_step(self, batch):
        """单步训练。"""
        texts = batch["state_texts"]
        target_actions = batch["optimal_actions"].to(self.device)
        target_transitions = batch["transition_probs"].to(self.device)
        target_q = batch["expected_returns"].to(self.device)
        num_valid = batch.get("num_valid_actions")
        action_texts = batch.get("action_texts")

        if num_valid is not None:
            num_actions = target_transitions.shape[1]
            valid_mask = torch.arange(num_actions, device=self.device).unsqueeze(0) < num_valid.unsqueeze(1)
        else:
            valid_mask = None

        call_kwargs = {}
        if action_texts is not None:
            call_kwargs["action_texts"] = action_texts
        if num_valid is not None:
            call_kwargs["num_valid_actions"] = num_valid

        result = self.model(texts, **call_kwargs)
        choice_logits = result["choice_probs"]
        predicted_transitions = result["transition_probs"]
        predicted_q = result["expected_returns"]

        losses = self.loss_fn(
            predicted_transitions=predicted_transitions,
            target_transitions=target_transitions,
            predicted_q_values=predicted_q,
            target_q_values=target_q,
            choice_logits=choice_logits,
            target_actions=target_actions,
            valid_mask=valid_mask,
        )

        return losses

    def train(self, train_loader, val_loader=None):
        """完整训练循环。"""
        self.logger.info("Starting RLCD-MDP training...")
        self.model.train()

        total_params, trainable_params = self.model.count_parameters()
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")

        self.optimizer.zero_grad()

        data_iter = iter(train_loader)
        total_batches = len(train_loader)

        for step in range(self.max_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            losses = self.train_step(batch)
            total_loss = losses["total_loss"]
            total_loss = total_loss / self.gradient_accumulation_steps
            total_loss.backward()

            if (step + 1) % self.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()

            self.global_step += 1

            if (step + 1) % 10 == 0:
                self.logger.info(
                    f"Step {step + 1}/{self.max_steps} | "
                    f"Loss: {total_loss.item() * self.gradient_accumulation_steps:.4f} | "
                    f"Calib: {losses['calib_loss'].item():.4f} | "
                    f"Q: {losses['q_value_loss'].item():.4f} | "
                    f"Dec: {losses['decision_loss'].item():.4f}"
                )

            if val_loader and (step + 1) % self.eval_every == 0:
                val_loss = self.evaluate(val_loader)
                self.model.train()

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best_model.pt")
                    self.logger.info(f"New best model! Val loss: {val_loss:.4f}")

            if (step + 1) % self.save_every == 0:
                self.save_checkpoint(f"checkpoint_step_{step + 1}.pt")

        self.save_checkpoint("final_model.pt")
        self.logger.info("Training completed!")
        return self.best_val_loss

    def evaluate(self, val_loader):
        """验证评估。"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                losses = self.train_step(batch)
                total_loss += losses["total_loss"].item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        self.logger.info(f"Validation loss: {avg_loss:.4f}")
        return avg_loss

    def save_checkpoint(self, filename):
        """保存检查点。"""
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / filename

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
                "global_step": self.global_step,
                "best_val_loss": self.best_val_loss,
            },
            path,
        )
        self.logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path):
        """加载检查点。"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.logger.info(f"Checkpoint loaded: {path}")
