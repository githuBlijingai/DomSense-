"""反馈数据转换与 DataLoader 构建。

将 DB 反馈行转换为 MDPDataPoint，构建可用于训练的 DataLoader。
"""
import json

from torch.utils.data import DataLoader

from src.data.synthetic import MDPDataPoint, MDPDataset
from src.utils.log import Logger
from .database import execute_query

logger = Logger().get_logger()


def feedback_row_to_datapoint(row: dict, num_outcomes: int = 2) -> MDPDataPoint:
    """将 DB 反馈行转换为 MDPDataPoint。

    Args:
        row: 反馈行字典（actions/transition_probs/expected_returns 可为 JSON 字符串或列表）
        num_outcomes: 结局数量

    Returns:
        MDPDataPoint 实例
    """
    actions = row["actions"]
    if isinstance(actions, str):
        actions = json.loads(actions)
    transition_probs = row["transition_probs"]
    if isinstance(transition_probs, str):
        transition_probs = json.loads(transition_probs)
    expected_returns = row["expected_returns"]
    if isinstance(expected_returns, str):
        expected_returns = json.loads(expected_returns)

    actions = list(actions)
    num_actions = len(actions)
    optimal_action = int(row["optimal_action"])

    # 防御性对齐：transition_probs / expected_returns 槽位数可能与 actions 不一致，
    # 截断/修正到与 actions 数量匹配，避免 collate / 训练时维度不匹配崩溃。
    transition_probs = [list(tp) for tp in (transition_probs or [])]
    expected_returns = list(expected_returns or [])
    if len(transition_probs) != num_actions:
        transition_probs = transition_probs[:num_actions]
        while len(transition_probs) < num_actions:
            transition_probs.append([1.0 / num_outcomes] * num_outcomes)
    if len(expected_returns) != num_actions:
        expected_returns = expected_returns[:num_actions]
        while len(expected_returns) < num_actions:
            expected_returns.append(0.0)
    if not (0 <= optimal_action < num_actions):
        optimal_action = num_actions - 1

    return MDPDataPoint(
        state_text=row["state_text"],
        actions=actions,
        optimal_action=optimal_action,
        transition_probs=transition_probs,
        expected_returns=expected_returns,
        num_outcomes=num_outcomes,
        domain=row.get("domain") or "general",
    )


def build_dataloader(
    feedbacks: list, batch_size: int = 16, num_outcomes: int = 2
) -> DataLoader:
    """从反馈列表构建 DataLoader。

    Args:
        feedbacks: 反馈行字典列表
        batch_size: 批大小
        num_outcomes: 结局数量

    Returns:
        DataLoader（collate_fn 使用 MDPDataset.collate_fn）
    """
    data_points = [
        feedback_row_to_datapoint(fb, num_outcomes=num_outcomes) for fb in feedbacks
    ]
    dataset = MDPDataset(data_points)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=MDPDataset.collate_fn,
    )


def count_unused() -> int:
    """查询 used_in_training=0 的反馈数量。"""
    rows = execute_query(
        "SELECT COUNT(*) AS cnt FROM feedbacks WHERE used_in_training = 0"
    )
    if rows:
        return int(rows[0]["cnt"])
    return 0


def get_unused_feedbacks(limit: int = None) -> list:
    """查询未用于训练的反馈（按创建时间升序）。

    Args:
        limit: 最多返回条数，None 表示全部

    Returns:
        反馈行字典列表
    """
    if limit is not None:
        sql = (
            "SELECT * FROM feedbacks WHERE used_in_training = 0 "
            "ORDER BY created_at ASC LIMIT ?"
        )
        return execute_query(sql, (int(limit),))
    sql = (
        "SELECT * FROM feedbacks WHERE used_in_training = 0 "
        "ORDER BY created_at ASC"
    )
    return execute_query(sql)


def mark_as_used(feedback_ids: list) -> None:
    """标记反馈为已用于训练。

    Args:
        feedback_ids: 反馈 id 列表
    """
    if not feedback_ids:
        return
    ids = [int(i) for i in feedback_ids]
    placeholders = ",".join(["?"] * len(ids))
    execute_query(
        f"UPDATE feedbacks SET used_in_training = 1 WHERE id IN ({placeholders})",
        tuple(ids),
    )
    logger.info(f"已标记 {len(ids)} 条反馈为已用于训练")


def validate_feedback(
    state_text: str,
    actions: list,
    optimal_action: int,
    transition_probs: list,
    expected_returns: list,
) -> list:
    """校验反馈数据，返回错误列表（空表示有效）。

    Args:
        state_text: 决策场景文本
        actions: 动作列表
        optimal_action: 最优动作索引（0-based）
        transition_probs: 转移概率矩阵 (num_actions, num_outcomes)
        expected_returns: 期望回报列表 (num_actions,)

    Returns:
        错误消息列表，空表示数据有效
    """
    errors = []
    # state_text 校验
    if not state_text or not str(state_text).strip():
        errors.append("state_text 不能为空")
    # actions 校验
    if not actions or not isinstance(actions, list):
        errors.append("actions 必须为非空列表")
    elif len(actions) < 1:
        errors.append("actions 至少需要 1 个")
    # optimal_action 校验
    if optimal_action is None or not isinstance(optimal_action, (int, float)):
        errors.append("optimal_action 必须为整数")
    else:
        optimal_action = int(optimal_action)
        if actions and (optimal_action < 0 or optimal_action >= len(actions)):
            errors.append(
                f"optimal_action 越界（应在 0-{len(actions) - 1}）"
            )
    # transition_probs 校验（可选：用户未填时传 None，由后端用模型预测填充）
    if transition_probs is not None:
        if not isinstance(transition_probs, list):
            errors.append("transition_probs 必须为二维列表")
        elif actions and len(transition_probs) != len(actions):
            errors.append(
                f"transition_probs 行数({len(transition_probs)})需等于 actions 数量({len(actions)})"
            )
        else:
            for i, tp in enumerate(transition_probs):
                if not isinstance(tp, list) or len(tp) < 1:
                    errors.append(f"transition_probs[{i}] 必须为非空列表")
                    continue
                total = sum(float(p) for p in tp)
                if abs(total - 1.0) > 0.05:
                    errors.append(
                        f"transition_probs[{i}] 概率之和应为 1.0，实际 {total:.4f}"
                    )
    # expected_returns 校验（可选：用户未填时传 None，由后端用模型预测填充）
    if expected_returns is not None:
        if not isinstance(expected_returns, list):
            errors.append("expected_returns 必须为列表")
        elif actions and len(expected_returns) != len(actions):
            errors.append(
                f"expected_returns 长度({len(expected_returns)})需等于 actions 数量({len(actions)})"
            )
    return errors
