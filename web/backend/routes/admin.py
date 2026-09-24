"""管理路由：CSV 上传、手动训练、训练记录查询、模型信息。"""
import csv
import io
import json
from datetime import datetime
from typing import List

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from .. import feedback_store
from ..database import execute_query
from ..model_manager import ModelManager
from ..schemas import (
    ModelInfo,
    TrainRequest,
    TrainResponse,
    TrainingRecord,
    UploadResponse,
)
from ..training_worker import TrainingWorker

router = APIRouter(prefix="/admin")


@router.post("/upload-csv", response_model=UploadResponse)
async def upload_csv(file: UploadFile = File(...)) -> UploadResponse:
    """上传 CSV 批量反馈。

    CSV 列：state_text, actions(JSON), optimal_action,
            transition_probs(JSON), expected_returns(JSON),
            predicted_choice(可空), source(可空)
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="仅支持 .csv 文件")

    content = await file.read()
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("gbk", errors="ignore")

    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise HTTPException(status_code=400, detail="CSV 文件为空或缺少表头")

    uploaded = 0
    errors = []
    created_at = datetime.now().isoformat()

    for line_no, row in enumerate(reader, start=2):
        try:
            state_text = (row.get("state_text") or "").strip()
            actions = json.loads(row.get("actions", "[]"))
            optimal_action = int(row.get("optimal_action", -1))
            transition_probs = json.loads(row.get("transition_probs", "[]"))
            expected_returns = json.loads(row.get("expected_returns", "[]"))
            pred_raw = row.get("predicted_choice")
            predicted_choice = (
                int(pred_raw) if pred_raw not in (None, "", "null") else None
            )
            source = (row.get("source") or "csv").strip() or "csv"
        except Exception as e:
            errors.append({"line": line_no, "error": f"解析失败: {e}"})
            continue

        err_msgs = feedback_store.validate_feedback(
            state_text,
            actions,
            optimal_action,
            transition_probs,
            expected_returns,
        )
        if err_msgs:
            errors.append({"line": line_no, "error": "; ".join(err_msgs)})
            continue

        execute_query(
            """INSERT INTO feedbacks
               (state_text, actions, optimal_action, transition_probs,
                expected_returns, predicted_choice, source, created_at,
                used_in_training)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                state_text,
                json.dumps(actions, ensure_ascii=False),
                int(optimal_action),
                json.dumps(transition_probs, ensure_ascii=False),
                json.dumps(expected_returns, ensure_ascii=False),
                predicted_choice,
                source,
                created_at,
                0,
            ),
        )
        uploaded += 1

    training_triggered = False
    if uploaded > 0:
        worker = TrainingWorker()
        training_triggered = worker.check_and_trigger()

    return UploadResponse(
        uploaded=uploaded,
        errors=errors,
        training_triggered=training_triggered,
    )


@router.post("/train", response_model=TrainResponse)
def train(req: TrainRequest) -> TrainResponse:
    """手动触发训练。

    先插入 training_records status='queued'，再 enqueue 任务。
    若队列满则回滚记录为 'failed'。
    """
    worker = TrainingWorker()
    if worker.is_busy():
        raise HTTPException(status_code=409, detail="已有训练任务在运行")

    run_name = req.run_name or (
        f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    created_at = datetime.now().isoformat()
    start_time = datetime.now().isoformat()

    rows = execute_query(
        """INSERT INTO training_records
           (run_name, trigger_source, num_samples, status,
            start_time, end_time, best_val_loss, checkpoint_path,
            feedback_ids, log, error_message, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            run_name,
            "manual",
            0,
            "queued",
            start_time,
            None,
            None,
            None,
            None,
            "",
            None,
            created_at,
        ),
    )
    training_id = rows[0]["lastrowid"] if rows else 0

    task = {
        "training_id": training_id,
        "run_name": run_name,
        "trigger_source": "manual",
        "max_steps": int(req.max_steps),
        "feedback_ids": req.feedback_ids,
    }
    ok = worker.enqueue(task)
    if not ok:
        execute_query(
            """UPDATE training_records
               SET status = ?, error_message = ?
               WHERE id = ?""",
            ("failed", "训练队列已满", training_id),
        )
        raise HTTPException(status_code=503, detail="训练队列已满")

    return TrainResponse(training_id=training_id, status="queued")


@router.get("/training-records", response_model=List[TrainingRecord])
def list_training_records(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> List[TrainingRecord]:
    """分页查询训练记录。"""
    offset = (page - 1) * page_size
    rows = execute_query(
        """SELECT id, run_name, trigger_source, status, num_samples,
                  best_val_loss, start_time, end_time, checkpoint_path
           FROM training_records
           ORDER BY id DESC
           LIMIT ? OFFSET ?""",
        (page_size, offset),
    )
    return [TrainingRecord(**row) for row in rows]


@router.get("/training-records/{record_id}", response_model=TrainingRecord)
def get_training_record(record_id: int) -> TrainingRecord:
    """单条训练记录详情。"""
    rows = execute_query(
        """SELECT id, run_name, trigger_source, status, num_samples,
                  best_val_loss, start_time, end_time, checkpoint_path
           FROM training_records
           WHERE id = ?""",
        (record_id,),
    )
    if not rows:
        raise HTTPException(status_code=404, detail="训练记录不存在")
    return TrainingRecord(**rows[0])


@router.delete("/training-records/{record_id}")
def delete_training_record(record_id: int) -> dict:
    """删除单条训练记录。"""
    execute_query("DELETE FROM training_records WHERE id = ?", (record_id,))
    return {"deleted": True, "id": record_id}


@router.get("/model/info", response_model=ModelInfo)
def model_info() -> ModelInfo:
    """模型信息。"""
    info = ModelManager().get_info()
    return ModelInfo(**info)
