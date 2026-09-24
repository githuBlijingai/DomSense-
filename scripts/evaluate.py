import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml

from src.models.system_one_model import SystemOneModel
from src.eval.calibration import CalibrationEvaluator
from src.eval.return_error import ReturnErrorEvaluator
from src.eval.benchmark import BenchmarkEvaluator
from src.data.builder import DataBuilder
from src.utils.log import Logger


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="RLCD-MDP 评测")
    parser.add_argument("--config", type=str, default="config/train_rlcd_mdp.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="模型检查点路径")
    parser.add_argument("--quick_test", action="store_true", help="快速测试模式")
    args = parser.parse_args()

    raw_config = load_config(args.config)
    base_config = load_config("config/base.yaml")
    config = {**base_config, **raw_config}
    for section in ["model", "training", "data", "loss", "eval"]:
        base_section = base_config.get(section, {})
        raw_section = raw_config.get(section, {})
        config[section] = {**base_section, **raw_section}
    model_config = config.get("model", {})
    data_config = config.get("data", {})
    train_config = config.get("training", {})

    full_config = {**model_config, **data_config, **train_config}

    if args.quick_test:
        full_config["num_synthetic_test"] = 50

    logger = Logger().get_logger()

    logger.info("=" * 50)
    logger.info("RLCD-MDP 评测")
    logger.info("=" * 50)

    logger.info("初始化模型...")
    model = SystemOneModel(model_config)

    if args.checkpoint:
        logger.info(f"加载检查点: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])

    logger.info("构建测试数据...")
    data_builder = DataBuilder(full_config)
    _, _, test_loader = data_builder.build_synthetic(
        full_config.get("num_synthetic_train", 5000),
        full_config.get("num_synthetic_val", 500),
        full_config.get("num_synthetic_test", 500),
    )
    logger.info(f"测试集: {len(test_loader.dataset)} 样本")

    model.eval()

    calib_eval = CalibrationEvaluator(config.get("eval", {}).get("num_bins", 10))
    return_eval = ReturnErrorEvaluator()
    bench_eval = BenchmarkEvaluator(model)

    print("\n=== 校准指标 ===")
    calib_results = calib_eval.evaluate_model_calibration(model, test_loader, device="cpu")
    for k, v in calib_results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n=== 回报预测误差 ===")
    return_results = return_eval.evaluate(model, test_loader, device="cpu")
    for k, v in return_results.items():
        print(f"  {k}: {v:.4f}")

    print("\n=== 延迟评估 ===")
    sample_texts = [dp.state_text for dp in test_loader.dataset.data_points[:10]]
    latency = bench_eval.measure_latency(sample_texts, num_runs=5)
    print(f"  平均延迟 ({len(sample_texts)}条): {latency:.2f} ms")

    logger.info("评测完成！")


if __name__ == "__main__":
    main()
