"""Pydantic 数据模型：请求与响应结构定义。"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """预测请求。"""

    state_text: str
    actions: List[str] = Field(default_factory=list)


class PredictResponse(BaseModel):
    """预测响应。"""

    request_id: str
    choice: int
    choice_label: str
    choice_probs: List[float]
    transition_probs: List[List[float]]
    expected_returns: List[float]
    expected_return_of_choice: float
    confidence: float
    compact_output: Dict[str, Any]
    model_version: int


class FeedbackRequest(BaseModel):
    """反馈请求。"""

    request_id: Optional[str] = None
    state_text: str
    actions: List[str]
    optimal_action: int
    transition_probs: Optional[List[List[float]]] = None
    expected_returns: Optional[List[float]] = None
    predicted_choice: Optional[int] = None
    source: str = "web"
    domain: str = "general"


class FeedbackResponse(BaseModel):
    """反馈响应。"""

    feedback_id: int
    accumulated_count: int
    will_trigger_training: bool
    next_training_in: int


class FeedbackStats(BaseModel):
    """反馈统计。"""

    total: int
    unused: int
    threshold: int
    next_training_in: int
    last_training_at: Optional[str] = None
    last_training_status: Optional[str] = None


class TrainRequest(BaseModel):
    """手动训练请求。"""

    feedback_ids: Optional[List[int]] = None
    max_steps: int = 200
    run_name: Optional[str] = None


class TrainResponse(BaseModel):
    """训练响应。"""

    training_id: int
    status: str


class TrainingRecord(BaseModel):
    """训练记录。"""

    id: int
    run_name: Optional[str] = None
    trigger_source: Optional[str] = None
    status: str
    num_samples: Optional[int] = None
    best_val_loss: Optional[float] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    checkpoint_path: Optional[str] = None


class ModelInfo(BaseModel):
    """模型信息。"""

    version: int
    checkpoint_path: str
    loaded_at: str
    best_val_loss: Optional[float] = None
    num_params: int
    trainable_params: int


class UploadResponse(BaseModel):
    """CSV 上传响应。"""

    uploaded: int
    errors: List[Dict[str, Any]]
    training_triggered: bool
