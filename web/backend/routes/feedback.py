"""反馈路由：POST /feedback + GET /stats。"""
import json
from datetime import datetime

from fastapi import APIRouter, HTTPException

from .. import feedback_store
from ..config import FEEDBACK_THRESHOLD
from ..database import execute_query
from ..model_manager import ModelManager
from ..schemas import FeedbackRequest, FeedbackResponse, FeedbackStats
from ..training_worker import TrainingWorker

router = APIRouter(prefix="/feedback")


@router.post("", response_model=FeedbackResponse)
def submit_feedback(req: FeedbackRequest) -> FeedbackResponse:
    """提交反馈：校验 → 填充缺失预测值 → 入库 → check_and_trigger → 响应。"""
    errors = feedback_store.validate_feedback(
        req.state_text,
        req.actions,
        req.optimal_action,
        req.transition_probs,
        req.expected_returns,
    )
    if errors:
        raise HTTPException(status_code=422, detail="; ".join(errors))

    # 用户未填转移概率/期望回报时，用模型当前预测值填充（作为弱标签）
    transition_probs = req.transition_probs
    expected_returns = req.expected_returns
    if transition_probs is None or expected_returns is None:
        manager = ModelManager()
        if manager.get_model() is not None:
            try:
                pred = manager.predict([req.state_text])
                if transition_probs is None:
                    transition_probs = pred["transition_probs"]
                if expected_returns is None:
                    expected_returns = pred["expected_returns"]
            except Exception as e:
                # 预测失败则回退到均匀占位
                n = len(req.actions)
                if transition_probs is None:
                    transition_probs = [[0.5, 0.5]] * n
                if expected_returns is None:
                    expected_returns = [0.5] * n
        # 对齐到真实动作数：模型预测可能输出 6 槽（无 actions 参数时），
        # 但 req.actions 只有 n 个 → 截断多余槽位
        n = len(req.actions)
        if transition_probs is not None and len(transition_probs) != n:
            transition_probs = transition_probs[:n]
        if expected_returns is not None and len(expected_returns) != n:
            expected_returns = expected_returns[:n]

    created_at = datetime.now().isoformat()
    domain = getattr(req, "domain", None) or "general"
    rows = execute_query(
        """INSERT INTO feedbacks
           (state_text, actions, optimal_action, transition_probs,
            expected_returns, predicted_choice, source, created_at,
            used_in_training, domain)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            req.state_text,
            json.dumps(req.actions, ensure_ascii=False),
            int(req.optimal_action),
            json.dumps(transition_probs, ensure_ascii=False),
            json.dumps(expected_returns, ensure_ascii=False),
            req.predicted_choice,
            req.source,
            created_at,
            0,
            domain,
        ),
    )
    feedback_id = rows[0]["lastrowid"] if rows else 0

    accumulated = feedback_store.count_unused()
    next_in = max(0, FEEDBACK_THRESHOLD - accumulated)
    will_trigger = False
    if accumulated >= FEEDBACK_THRESHOLD:
        worker = TrainingWorker()
        triggered = worker.check_and_trigger()
        will_trigger = bool(triggered)

    return FeedbackResponse(
        feedback_id=feedback_id,
        accumulated_count=accumulated,
        will_trigger_training=will_trigger,
        next_training_in=next_in,
    )


@router.get("/stats", response_model=FeedbackStats)
def feedback_stats() -> FeedbackStats:
    """反馈统计。"""
    total_rows = execute_query("SELECT COUNT(*) AS cnt FROM feedbacks")
    total = int(total_rows[0]["cnt"]) if total_rows else 0
    unused = feedback_store.count_unused()
    last_rows = execute_query(
        "SELECT status, end_time FROM training_records ORDER BY id DESC LIMIT 1"
    )
    last_status = last_rows[0]["status"] if last_rows else None
    last_at = last_rows[0]["end_time"] if last_rows else None
    return FeedbackStats(
        total=total,
        unused=unused,
        threshold=FEEDBACK_THRESHOLD,
        next_training_in=max(0, FEEDBACK_THRESHOLD - unused),
        last_training_at=last_at,
        last_training_status=last_status,
    )
