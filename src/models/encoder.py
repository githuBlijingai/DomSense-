import torch
import torch.nn as nn


class LightweightTextEncoder(nn.Module):
    """轻量级字符/词元编码器（离线 fallback）。

    当无法下载预训练模型时，使用基于字符嵌入的简单编码器，
    保证整个流程在离线 CPU 环境也能运行。
    """

    def __init__(self, hidden_dim=128, vocab_dim=256, max_len=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_dim, hidden_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 2, batch_first=True
        )
        self.encoder_transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

    def tokenize_texts(self, texts):
        """将文本转换为字符 ID 序列。"""
        batch_sequences = []
        for text in texts:
            byte_ids = [min(b, 255) for b in text.encode("utf-8", errors="ignore")[: self.max_len]]
            pad_len = self.max_len - len(byte_ids)
            byte_ids = byte_ids + [0] * pad_len
            batch_sequences.append(byte_ids)
        return torch.tensor(batch_sequences, dtype=torch.long)

    def forward(self, texts):
        """编码文本。

        Args:
            texts: List[str]

        Returns:
            embeddings: (batch, hidden_dim)
        """
        device = next(self.parameters()).device
        tokens = self.tokenize_texts(texts).to(device)
        token_mask = (tokens != 0).unsqueeze(-1).float()

        embedded = self.embedding(tokens)
        embedded = embedded * token_mask

        features = self.encoder_transformer(embedded)

        pooled = features.mean(dim=1)
        norm = torch.norm(pooled, p=2, dim=-1, keepdim=True).clamp(min=1e-6)
        pooled = pooled / norm

        return pooled


class LLMEncoder(nn.Module):
    """LLM 编码器封装。

    将开源小模型（如 distilbert-base-uncased）作为语义编码器，
    支持冻结或 LoRA 微调。若无法联网下载预训练模型，
    会自动回退到内置的轻量级编码器（离线可用）。
    """

    def __init__(self, config):
        super().__init__()
        self.encoder_name = config.get("encoder_name", "distilbert-base-uncased")
        self.freeze = config.get("encoder_freeze", True)
        self.use_lora = config.get("use_lora", True)
        self.hidden_dim = config.get("hidden_dim", 768)
        self.backend = "none"

        self.encoder = self._build_encoder(config)

    def _build_encoder(self, config):
        import os

        HF_HUB_OFFLINE = config.get("hf_offline", False)

        if HF_HUB_OFFLINE:
            return self._build_lightweight(config)

        try:
            from transformers import AutoModel, AutoTokenizer
            from peft import LoraConfig, get_peft_model, TaskType

            self.tokenizer = AutoTokenizer.from_pretrained(self.encoder_name)
            encoder = AutoModel.from_pretrained(self.encoder_name)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            if self.freeze:
                for param in encoder.parameters():
                    param.requires_grad = False

            if self.use_lora and not self.freeze:
                lora_config = LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=config.get("lora_r", 8),
                    lora_alpha=config.get("lora_alpha", 16),
                    lora_dropout=config.get("lora_dropout", 0.1),
                    target_modules=["q_lin", "k_lin", "v_lin", "out_lin"],
                )
                encoder = get_peft_model(encoder, lora_config)

            self.output_projection = nn.Linear(
                encoder.config.hidden_size, self.hidden_dim
            )
            self.backend = "transformers"
            return encoder

        except Exception as e:
            print(
                f"[Encoder] Failed to load pretrained model '{self.encoder_name}': {e}\n"
                f"[Encoder] Falling back to lightweight local encoder (offline mode)."
            )
            return self._build_lightweight(config)

    def _build_lightweight(self, config):
        """构建轻量级离线编码器。"""
        embed_dim = config.get("embed_dim", min(128, self.hidden_dim))
        self.backend = "lightweight"
        encoder = LightweightTextEncoder(
            hidden_dim=embed_dim, vocab_dim=256, max_len=128
        )
        self.output_projection = nn.Linear(embed_dim, self.hidden_dim)
        self.encode_text = encoder.forward
        return encoder

    def forward(self, texts):
        """编码输入文本为语义嵌入。

        Args:
            texts: List[str] 输入文本

        Returns:
            embeddings: (batch, hidden_dim) 语义嵌入
        """
        if self.backend == "lightweight":
            raw = self._pylight_encode(texts)
            return self.output_projection(raw)

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.encoder.device) for k, v in inputs.items()}
        with torch.set_grad_enabled(not self.freeze):
            outputs = self.encoder(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings = self.output_projection(embeddings)
        return embeddings

    def _pylight_encode(self, texts):
        """内部轻量编码（保持 batch_first 等）。"""
        if isinstance(texts, str):
            texts = [texts]
        return self.encoder(texts)
