import torch
from torch.utils.data import DataLoader, random_split

from .synthetic import SyntheticDataGenerator, MDPDataset
from .external import ExternalDataset


class DataBuilder:
    """数据构建器：整合合成数据和公开数据。"""

    def __init__(self, config):
        self.config = config
        self.synthetic_generator = SyntheticDataGenerator(config)

    def build_synthetic(self, num_train, num_val, num_test):
        """构建合成数据集。

        Args:
            num_train: int
            num_val: int
            num_test: int

        Returns:
            train_loader, val_loader, test_loader
        """
        train_dataset = self.synthetic_generator.generate_dataset(num_train)
        val_dataset = self.synthetic_generator.generate_dataset(num_val)
        test_dataset = self.synthetic_generator.generate_dataset(num_test)

        batch_size = self.config.get("batch_size", 16)

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
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=MDPDataset.collate_fn,
        )

        return train_loader, val_loader, test_loader

    def build_external(self, name="ag_news", max_samples=1000):
        """构建外部数据集 DataLoader。

        Args:
            name: 数据集名称
            max_samples: 最大样本数

        Returns:
            DataLoader
        """
        dataset = ExternalDataset.load_dataset_by_name(
            name, split="train", max_samples=max_samples
        )
        batch_size = self.config.get("batch_size", 16)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=MDPDataset.collate_fn,
        )

    def build_mixed(self, synthetic_size=2000, external_name="ag_news", external_size=500):
        """构建混合数据集。"""
        from torch.utils.data import ConcatDataset

        synthetic_dataset = self.synthetic_generator.generate_dataset(synthetic_size)
        external_dataset = ExternalDataset.load_dataset_by_name(
            external_name, split="train", max_samples=external_size
        )

        mixed_dataset = ConcatDataset([synthetic_dataset, external_dataset])
        batch_size = self.config.get("batch_size", 16)

        return DataLoader(
            mixed_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
