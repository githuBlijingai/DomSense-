"""训练工作线程：后台异步训练 + 自动触发。

单例模式。任务队列 + daemon 线程。
自动触发：当未用反馈数量 >= 阈值且无运行中任务时入队自动训练。
"""
import json
import os
import queue
import threading
import traceback
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from src.data.synthetic import MDPDataset
from src.models.system_one_model import SystemOneModel
from src.training.rlcd_mdp_trainer import RLCDMDPTrainer
from src.utils.log import Logger

from . import config as web_config
from . import feedback_store
from .config import (
    FEEDBACK_THRESHOLD,
    OUTPUT_DIR,
    TRAIN_EVAL_EVERY,
    TRAIN_MAX_STEPS,
    TRAIN_SAVE_EVERY,
    load_merged_config,
)
from .database import execute_query
from .model_manager import ModelManager

logger = Logger().get_logger()


class TrainingWorker:
    """训练工作单例。

    职责：
    - 维护任务队列（maxsize=4）
    - daemon 线程循环消费任务
    - 每个任务：构建 DataLoader → 80/20 划分 → 新模型加载 checkpoint →
      训练 → 热替换 → 标记反馈已用 → 更新训练记录
    - check_and_trigger：自动触发条件检查
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
        self.task_queue = queue.Queue(maxsize=4)
        self._stop = threading.Event()
        self._thread = None
        # 运行状态锁
        self._busy_lock = threading.Lock()
        self._running = False
        # 防止重复入队自动任务
        self._auto_pending = False

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def start(self) -> None:
        """启动 daemon 工作线程。"""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="training-worker"
        )
        self._thread.start()
        logger.info("训练工作线程已启动")

    def stop(self) -> None:
        """停止工作线程。"""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        logger.info("训练工作线程已停止")

    # ------------------------------------------------------------------
    # 队列操作
    # ------------------------------------------------------------------
    def enqueue(self, task: dict) -> bool:
        """入队任务，成功返回 True，队列满返回 False。"""
        try:
            self.task_queue.put_nowait(task)
            logger.info(
                f"训练任务已入队: run_name={task.get('run_name', 'auto')}, "
                f"source={task.get('trigger_source', 'manual')}"
            )
            return True
        except queue.Full:
            logger.warning("训练任务队列已满，入队失败")
            return False

    def is_busy(self) -> bool:
        """是否有运行中的训练任务。"""
        with self._busy_lock:
            return self._running

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    def _run(self) -> None:
        """循环取任务执行。"""
        while not self._stop.is_set():
            try:
                task = self.task_queue.get(timeout=1)
            except queue.Empty:
                continue
            self._execute(task)

    # ------------------------------------------------------------------
    # 执行单个训练任务
    # ------------------------------------------------------------------
    def _execute(self, task: dict) -> None:
        """执行单个训练任务。

        步骤：
        1. 插入/更新 training_records status='running'
        2. 查询未用反馈（或指定 feedback_ids）
        3. feedback_store.build_dataloader
        4. 80/20 train/val split
        5. new_model = SystemOneModel(model_config); load 当前 checkpoint
        6. RLCDMDPTrainer(new_model, flat_train_config)
        7. trainer.train(train_loader, val_loader)
        8. model_manager.swap_model(new_model, best_ckpt_path, best_val_loss)
        9. mark_as_used(feedback_ids)
        10. 更新 training_records status='completed'（异常时 status='failed'）
        """
        with self._busy_lock:
            self._running = True
        # 清除自动挂起标记（任务开始执行，允许后续自动触发）
        self._auto_pending = False

        run_name = task.get("run_name") or (
            f"web_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        trigger_source = task.get("trigger_source", "manual")
        max_steps = int(task.get("max_steps", TRAIN_MAX_STEPS))
        feedback_ids = task.get("feedback_ids")
        num_outcomes = int(task.get("num_outcomes", 2))
        batch_size = int(task.get("batch_size", 16))

        log_lines = []

        def append_log(line: str) -> None:
            log_lines.append(line)
            logger.info(line)

        start_time = datetime.now().isoformat()
        created_at = datetime.now().isoformat()

        # 1. 训练记录初始化
        training_id = task.get("training_id")
        if training_id is None:
            # 自动任务：插入新记录
            rows = execute_query(
                """INSERT INTO training_records
                   (run_name, trigger_source, num_samples, status,
                    start_time, end_time, best_val_loss, checkpoint_path,
                    feedback_ids, log, error_message, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_name,
                    trigger_source,
                    0,
                    "running",
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
            training_id = rows[0]["lastrowid"] if rows else None
        else:
            # 手动任务：更新已存在的 'queued' 记录为 'running'
            execute_query(
                """UPDATE training_records
                   SET status = ?, start_time = ?
                   WHERE id = ?""",
                ("running", start_time, training_id),
            )

        append_log(
            f"训练任务开始: id={training_id}, run_name={run_name}, "
            f"source={trigger_source}, max_steps={max_steps}"
        )

        try:
            # 2. 查询反馈
            if feedback_ids:
                placeholders = ",".join(["?"] * len(feedback_ids))
                feedbacks = execute_query(
                    f"SELECT * FROM feedbacks WHERE id IN ({placeholders})",
                    tuple(int(i) for i in feedback_ids),
                )
            else:
                feedbacks = feedback_store.get_unused_feedbacks()

            if len(feedbacks) < 2:
                raise ValueError(
                    f"可用反馈不足（{len(feedbacks)} 条），至少需要 2 条"
                )
            append_log(f"使用反馈样本数: {len(feedbacks)}")

            # 3. 构建 DataLoader（先取全部，再划分）
            full_loader = feedback_store.build_dataloader(
                feedbacks,
                batch_size=batch_size,
                num_outcomes=num_outcomes,
            )
            dataset = full_loader.dataset
            n = len(dataset)
            # 4. 80/20 train/val split
            n_val = max(1, n // 5)
            n_train = n - n_val
            train_dataset = MDPDataset(dataset.data_points[:n_train])
            val_dataset = MDPDataset(dataset.data_points[n_train:])
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=MDPDataset.collate_fn,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=MDPDataset.collate_fn,
            )
            append_log(f"训练集: {n_train}，验证集: {n_val}")

            # 5. 新模型 + 加载当前 checkpoint
            merged = load_merged_config()
            model_config = dict(merged.get("model", {}))
            train_section = dict(merged.get("training", {}))
            loss_section = dict(merged.get("loss", {}))
            data_section = dict(merged.get("data", {}))

            new_model = SystemOneModel(model_config)
            ckpt_path = task.get("checkpoint_path") or web_config.CHECKPOINT_PATH
            if ckpt_path and os.path.exists(ckpt_path):
                try:
                    ckpt = torch.load(ckpt_path, map_location="cpu")
                    new_model.load_state_dict(ckpt["model_state_dict"])
                    append_log(f"已加载初始 checkpoint: {ckpt_path}")
                except Exception as e:
                    append_log(f"加载初始 checkpoint 失败（使用随机初始化）: {e}")
            else:
                append_log(
                    f"初始 checkpoint 不存在: {ckpt_path}，使用随机初始化"
                )

            # 6. 构建训练配置（扁平化，覆盖 web 层参数）
            run_output_dir = os.path.join(OUTPUT_DIR, run_name)
            os.makedirs(run_output_dir, exist_ok=True)
            flat_config = {}
            flat_config.update(model_config)
            flat_config.update(train_section)
            flat_config.update(loss_section)
            flat_config.update(data_section)
            flat_config["max_steps"] = max_steps
            flat_config["eval_every"] = TRAIN_EVAL_EVERY
            flat_config["save_every"] = TRAIN_SAVE_EVERY
            flat_config["output_dir"] = run_output_dir

            # 7. 训练
            trainer = RLCDMDPTrainer(new_model, flat_config)
            best_val_loss = trainer.train(train_loader, val_loader)
            append_log(
                f"训练完成: best_val_loss={best_val_loss}"
            )

            # 8. 热替换模型
            best_ckpt_path = os.path.join(run_output_dir, "best_model.pt")
            if not os.path.exists(best_ckpt_path):
                best_ckpt_path = os.path.join(run_output_dir, "final_model.pt")
            model_manager = ModelManager()
            model_manager.swap_model(new_model, best_ckpt_path, best_val_loss)
            append_log(f"已热替换模型: {best_ckpt_path}")

            # 9. 标记反馈为已用
            used_ids = [fb["id"] for fb in feedbacks]
            feedback_store.mark_as_used(used_ids)

            # 10. 更新训练记录 status='completed'
            end_time = datetime.now().isoformat()
            execute_query(
                """UPDATE training_records
                   SET status = ?, num_samples = ?, end_time = ?,
                       best_val_loss = ?, checkpoint_path = ?,
                       feedback_ids = ?, log = ?
                   WHERE id = ?""",
                (
                    "completed",
                    len(feedbacks),
                    end_time,
                    float(best_val_loss) if best_val_loss is not None else None,
                    best_ckpt_path,
                    json.dumps(used_ids),
                    "\n".join(log_lines),
                    training_id,
                ),
            )
            append_log("训练任务完成并已记录")
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            tb = traceback.format_exc()
            append_log(f"训练失败: {err}")
            append_log(tb)
            end_time = datetime.now().isoformat()
            try:
                execute_query(
                    """UPDATE training_records
                       SET status = ?, end_time = ?, error_message = ?, log = ?
                       WHERE id = ?""",
                    (
                        "failed",
                        end_time,
                        err + "\n" + tb,
                        "\n".join(log_lines),
                        training_id,
                    ),
                )
            except Exception as update_err:
                logger.error(f"更新失败记录时出错: {update_err}")
        finally:
            with self._busy_lock:
                self._running = False

    # ------------------------------------------------------------------
    # 自动触发
    # ------------------------------------------------------------------
    def check_and_trigger(self) -> bool:
        """检查 unused 数量 >= threshold 且无 running 任务则 enqueue。

        使用 _auto_pending 标记防止重复入队自动任务。

        Returns:
            是否成功触发（入队）自动训练
        """
        if self.is_busy():
            return False
        if self._auto_pending:
            return False
        unused = feedback_store.count_unused()
        if unused < FEEDBACK_THRESHOLD:
            return False
        run_name = f"auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        task = {
            "run_name": run_name,
            "trigger_source": "auto",
            "max_steps": TRAIN_MAX_STEPS,
        }
        self._auto_pending = True
        ok = self.enqueue(task)
        if not ok:
            self._auto_pending = False
        return ok
