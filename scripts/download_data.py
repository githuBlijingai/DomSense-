import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.external import ExternalDataset


def main():
    print("下载 AG News 数据集...")
    dataset = ExternalDataset.load_ag_news(split="train", max_samples=100)
    print(f"AG News 转换完成: {len(dataset)} 样本")

    print("下载 TREC 数据集...")
    dataset = ExternalDataset.load_trec(split="train", max_samples=100)
    print(f"TREC 转换完成: {len(dataset)} 样本")


if __name__ == "__main__":
    main()
