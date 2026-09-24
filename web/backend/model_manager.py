"""模型管理器：单例，负责模型加载、推理与热替换。

线程安全：使用 RLock 保护模型引用的读写。
推理使用 torch.no_grad()。
"""
import threading
from datetime import datetime

import torch

from src.models.system_one_model import SystemOneModel
from src.utils.log import Logger
from .config import CHECKPOINT_PATH

logger = Logger().get_logger()


class ModelManager:
    """模型管理单例。

    职责：
    - 加载初始模型（从 checkpoint）
    - 提供推理接口（forward + generate_compact_output）
    - 热替换模型（训练完成后 swap，version + 1）
    - 暴露模型版本/路径/loss/参数量
    """

    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._lock = threading.RLock()
        self._model = None
        self.version = 0
        self.checkpoint_path = ""
        self.loaded_at = ""
        self.best_val_loss = None

    def load_initial(self, checkpoint_path: str, model_config: dict) -> None:
        """加载初始模型。

        Args:
            checkpoint_path: checkpoint 文件路径
            model_config: 模型配置字典（含 hidden_dim / num_actions 等）
        """
        with self._lock:
            model = SystemOneModel(model_config)
            loaded = False
            try:
                if checkpoint_path and _file_exists(checkpoint_path):
                    ckpt = torch.load(checkpoint_path, map_location="cpu")
                    use_domain = model_config.get("use_domain_experts", False)
                    if use_domain:
                        # 软迁移：启用领域 MoE 时，允许旧 checkpoint（共享架构）
                        # 只加载重叠的共享权重，新的领域专家键保持随机初始化。
                        missing, unexpected = model.load_state_dict(
                            ckpt["model_state_dict"], strict=False
                        )
                        if missing:
                            logger.info(
                                f"领域 MoE 软迁移：{len(missing)} 个新键随机初始化"
                            )
                    else:
                        model.load_state_dict(ckpt["model_state_dict"])
                    self.best_val_loss = ckpt.get("best_val_loss")
                    loaded = True
                    logger.info(f"已加载 checkpoint: {checkpoint_path}")
            except Exception as e:
                logger.warning(
                    f"加载 checkpoint 失败: {e}，使用随机初始化模型"
                )
            model.eval()
            self._model = model
            self.checkpoint_path = checkpoint_path
            self.loaded_at = datetime.now().isoformat()
            self.version = 1
            total, trainable = model.count_parameters()
            logger.info(
                f"模型已加载: version={self.version}, "
                f"params={total:,}, trainable={trainable:,}, loaded={loaded}"
            )

    def predict(self, texts: list, actions: list = None) -> dict:
        """推理：调用 model.forward + generate_compact_output。

        领域由模型门控网络从场景语义自动路由，无需外部传入 domain。

        Args:
            texts: 输入文本列表
            actions: 可选动作列表（仅用于上下文，模型内部用自身 num_actions）

        Returns:
            结果字典，含 choice / choice_label / choice_probs /
            transition_probs / expected_returns / expected_return_of_choice /
            confidence / compact_output / model_version
        """
        with self._lock:
            if self._model is None:
                raise RuntimeError("模型尚未加载")
            call_kwargs = {}
            if actions:
                call_kwargs["action_texts"] = [actions]
                call_kwargs["num_valid_actions"] = torch.tensor([len(actions)])
            with torch.no_grad():
                outputs = self._model.forward(texts, **call_kwargs)
                compact = self._model.generate_compact_output(
                    texts,
                    action_texts=call_kwargs.get("action_texts"),
                    num_valid_actions=call_kwargs.get("num_valid_actions"),
                )
            idx = 0
            choice = int(outputs["choices"][idx].item())
            choice_probs = (
                outputs["choice_probs"][idx].detach().cpu().tolist()
            )
            transition_probs = (
                outputs["transition_probs"][idx].detach().cpu().tolist()
            )
            expected_returns = (
                outputs["expected_returns"][idx].detach().cpu().tolist()
            )
            conf_tensor = outputs["confidences"]
            # confidences 形状 (batch,)，取第 0 个样本
            confidence = float(conf_tensor[idx].item())
            expected_return_of_choice = float(
                outputs["expected_returns"][idx, choice].item()
            )
            return {
                "choice": choice,
                "choice_label": f"选项{choice + 1}",
                "choice_probs": choice_probs,
                "transition_probs": transition_probs,
                "expected_returns": expected_returns,
                "expected_return_of_choice": expected_return_of_choice,
                "confidence": confidence,
                "compact_output": compact[0] if compact else {},
                "model_version": self.version,
            }

    def swap_model(self, new_model, checkpoint_path: str, best_val_loss) -> None:
        """热替换模型引用，version + 1。

        Args:
            new_model: 已训练好的新模型实例
            checkpoint_path: 新模型对应的 checkpoint 路径
            best_val_loss: 最佳验证损失
        """
        with self._lock:
            new_model.eval()
            self._model = new_model
            self.checkpoint_path = checkpoint_path
            self.best_val_loss = best_val_loss
            self.loaded_at = datetime.now().isoformat()
            self.version += 1
            logger.info(
                f"模型已热替换: version={self.version}, "
                f"checkpoint={checkpoint_path}, loss={best_val_loss}"
            )

    def get_info(self) -> dict:
        """返回模型版本/路径/loss/参数量。"""
        with self._lock:
            if self._model is None:
                return {
                    "version": self.version,
                    "checkpoint_path": self.checkpoint_path,
                    "loaded_at": self.loaded_at,
                    "best_val_loss": self.best_val_loss,
                    "num_params": 0,
                    "trainable_params": 0,
                }
            total, trainable = self._model.count_parameters()
            return {
                "version": self.version,
                "checkpoint_path": self.checkpoint_path,
                "loaded_at": self.loaded_at,
                "best_val_loss": self.best_val_loss,
                "num_params": total,
                "trainable_params": trainable,
            }

    def get_model(self):
        """返回当前模型引用（可能为 None）。"""
        with self._lock:
            return self._model


def _file_exists(path: str) -> bool:
    """安全检查文件是否存在。"""
    import os

    try:
        return os.path.exists(path)
    except Exception:
        return False
