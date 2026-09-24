import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml
import time

from src.models.system_one_model import SystemOneModel
from src.training.rlcd_mdp_trainer import RLCDMDPTrainer
from src.data.builder import DataBuilder
from src.utils.log import Logger


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="RLCD-MDP 训练")
    parser.add_argument("--config", type=str, default="config/train_rlcd_mdp.yaml")
    parser.add_argument("--use_external", action="store_true", help="是否使用外部数据集")
    parser.add_argument("--external_name", type=str, default="ag_news")
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
    train_config = config.get("training", {})
    data_config = config.get("data", {})

    config = {**model_config, **train_config, **data_config}

    if args.quick_test:
        config["num_synthetic_train"] = 200
        config["num_synthetic_val"] = 50
        config["num_synthetic_test"] = 50
        config["max_steps"] = 20
        config["eval_every"] = 5
        config["save_every"] = 100

    logger = Logger(log_dir=config.get("output_dir", "./outputs")).get_logger()

    logger.info("=" * 50)
    logger.info("RLCD-MDP 训练开始")
    logger.info(f"模型编码器: {model_config.get('encoder_name', 'distilbert-base-uncased')}")
    logger.info(f"Bellman 迭代步数: {model_config.get('num_bellman_steps', 4)}")
    logger.info("=" * 50)

    logger.info("初始化模型...")
    model = SystemOneModel(model_config)

    total_params, trainable_params = model.count_parameters()
    logger.info(f"总参数量: {total_params:,}")
    logger.info(f"可训练参数量: {trainable_params:,}")

    logger.info("构建数据...")
    data_builder = DataBuilder(config)
    train_loader, val_loader, test_loader = data_builder.build_synthetic(
        config.get("num_synthetic_train", 5000),
        config.get("num_synthetic_val", 500),
        config.get("num_synthetic_test", 500),
    )
    logger.info(
        f"训练集: {len(train_loader.dataset)} 样本, "
        f"验证集: {len(val_loader.dataset)} 样本, "
        f"测试集: {len(test_loader.dataset)} 样本"
    )

    trainer = RLCDMDPTrainer(model, config)
    trainer.train(train_loader, val_loader)

    logger.info("训练完成！")


if __name__ == "__main__":
    main()
