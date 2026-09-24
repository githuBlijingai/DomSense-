"""Web 层配置：路径常量与合并配置加载。

所有路径为相对路径（相对于项目根目录 D:\\Jev + MDP）。
离线环境，不依赖网络。
"""
import os

import yaml

# SQLite 数据库路径
DB_PATH = "data/feedback.db"

# 上传 CSV 存放目录
UPLOAD_DIR = "data/uploaded_csv"

# 初始模型 checkpoint 路径
CHECKPOINT_PATH = "outputs/rlcd_mdp_run1/best_model.pt"

# Web 训练输出目录
OUTPUT_DIR = "outputs/web_runs"

# 积累多少条反馈触发自动训练
FEEDBACK_THRESHOLD = 50

# 自动训练步数
TRAIN_MAX_STEPS = 200

# 自动训练评估间隔
TRAIN_EVAL_EVERY = 20

# 自动训练保存间隔
TRAIN_SAVE_EVERY = 50

# 服务监听地址与端口
HOST = "0.0.0.0"
PORT = 8000

# 合并配置缓存
_CONFIG_CACHE = None


def _deep_merge(base: dict, override: dict) -> dict:
    """深度合并两个字典：override 覆盖 base，嵌套字典递归合并。"""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_merged_config(force_reload: bool = False) -> dict:
    """加载 base.yaml + train_rlcd_mdp.yaml，深度合并。

    返回带 section 的合并字典（model / training / data / loss / eval）。
    train_rlcd_mdp.yaml 中的 defaults 字段（列表）会被忽略，仅用于参考。

    Args:
        force_reload: 是否强制重新读取（忽略缓存）

    Returns:
        合并后的配置字典
    """
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and not force_reload:
        return _CONFIG_CACHE

    base_path = os.path.join("config", "base.yaml")
    train_path = os.path.join("config", "train_rlcd_mdp.yaml")

    with open(base_path, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f) or {}
    with open(train_path, "r", encoding="utf-8") as f:
        train = yaml.safe_load(f) or {}

    # 移除 train 中的 defaults（仅是 OmegaConf 风格引用，不参与运行时合并）
    train.pop("defaults", None)

    merged = _deep_merge(base, train)
    _CONFIG_CACHE = merged
    return merged
