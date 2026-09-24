"""预测路由：POST /predict。"""
import uuid

from fastapi import APIRouter, HTTPException

from ..model_manager import ModelManager
from ..schemas import PredictRequest, PredictResponse

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """预测接口。

    - actions 为空时使用默认 ["选项1","选项2","选项3"]
    - 生成 request_id (uuid4)
    """
    manager = ModelManager()
    if manager.get_model() is None:
        raise HTTPException(status_code=503, detail="模型尚未加载")
    # actions 仅作上下文保留，模型内部使用自身 num_actions
    actions = req.actions if req.actions else ["选项1", "选项2", "选项3"]
    try:
        # 领域由模型自动路由，无需传入 domain
        result = manager.predict([req.state_text], actions=actions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理失败: {e}")
    request_id = str(uuid.uuid4())
    return PredictResponse(
        request_id=request_id,
        choice=result["choice"],
        choice_label=result["choice_label"],
        choice_probs=result["choice_probs"],
        transition_probs=result["transition_probs"],
        expected_returns=result["expected_returns"],
        expected_return_of_choice=result["expected_return_of_choice"],
        confidence=result["confidence"],
        compact_output=result["compact_output"],
        model_version=result["model_version"],
    )
