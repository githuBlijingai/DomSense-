"""模板路由：CSV 下载与字段说明。"""
import io

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

router = APIRouter()

# CSV 模板表头（列名即字段名）
_CSV_HEADER = (
    "state_text,actions,optimal_action,transition_probs,"
    "expected_returns,predicted_choice,source\n"
)

# 一行示例数据（可直接被解析）
_CSV_SAMPLE = (
    "用户需要选择最佳的方案来处理技术选型问题。当前选项：1. 方案A；2. 方案B；3. 方案C,"
    '["方案A","方案B","方案C"],'
    "1,"
    "[[0.7,0.3],[0.4,0.6],[0.5,0.5]],"
    "[5.5,7.2,6.0],"
    ",csv\n"
)

# 字段说明
_GUIDE = {
    "fields": [
        {
            "name": "state_text",
            "type": "string",
            "required": True,
            "desc": "决策场景文本",
        },
        {
            "name": "actions",
            "type": "json list[str]",
            "required": True,
            "desc": '可选动作列表，JSON 字符串，如 ["方案A","方案B","方案C"]',
        },
        {
            "name": "optimal_action",
            "type": "int",
            "required": True,
            "desc": "最优动作索引（0-based）",
        },
        {
            "name": "transition_probs",
            "type": "json list[list[float]]",
            "required": True,
            "desc": (
                "每动作的转移概率矩阵，JSON 字符串，"
                "行数=动作数，每行概率和=1.0，"
                "如 [[0.7,0.3],[0.4,0.6]]"
            ),
        },
        {
            "name": "expected_returns",
            "type": "json list[float]",
            "required": True,
            "desc": "每动作的期望回报，JSON 字符串，长度=动作数",
        },
        {
            "name": "predicted_choice",
            "type": "int|null",
            "required": False,
            "desc": "模型预测的选择索引（可空）",
        },
        {
            "name": "source",
            "type": "string",
            "required": False,
            "desc": "数据来源，默认 csv",
        },
    ],
    "notes": [
        "actions / transition_probs / expected_returns 需用 JSON 字符串表示",
        "transition_probs 行数应等于 actions 数量，每行概率和为 1.0",
        "expected_returns 长度应等于 actions 数量",
        "optimal_action 为 0-based 索引，必须在 actions 范围内",
    ],
}


@router.get("/template/annotation.csv")
def download_annotation_csv():
    """下载标注 CSV 模板（含表头 + 一行示例）。"""
    content = _CSV_HEADER + _CSV_SAMPLE
    return StreamingResponse(
        iter([content]),
        media_type="text/csv",
        headers={
            "Content-Disposition": (
                "attachment; filename=annotation_template.csv"
            )
        },
    )


@router.get("/template/annotation-guide")
def annotation_guide():
    """返回字段说明 JSON。"""
    return _GUIDE
