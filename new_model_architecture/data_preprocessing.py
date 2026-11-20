"""
Contains functions to preprocess the data for training, so main isn't very cluttered
"""
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.loader.dataloader import DataLoader

from new_model_architecture.data.dataloader import (
    SionnaMultiSceneDataset,
    SnapshotDataset,
)
from new_model_architecture.settings import settings
import os

def _load_dataset(filepaths: list[str |Path])-> SionnaMultiSceneDataset:
    """
    Load the dataset from the given filepaths.

    Args:
        filepaths (list): List of file paths to load the dataset from.

    Returns:
        SionnaMultiSceneDataset: The loaded dataset.
    """
    return SionnaMultiSceneDataset(filepaths=filepaths)

def load_dataloader_v1(
    train_dataset_filepaths: list[str | Path],
    test_dataset_filepaths: list[str | Path],
    batch_size: int = settings.BATCH_SIZE,
    train_val_split: float = 0.8,
    seed: int = 45234,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load the Dataloaders for train, val, and test for a "Row Based Dataset".
    The dataset splits each episode of moving jammer into individual rows.
    """

    train_val_dataset: SionnaMultiSceneDataset = _load_dataset(train_dataset_filepaths)
    test_dataset: SionnaMultiSceneDataset = _load_dataset(test_dataset_filepaths)


    # Split before expanding into snapshots
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_val_dataset,
        [int(len(train_val_dataset) * 0.8), len(train_val_dataset) - int(len(train_val_dataset) * 0.8)],
        generator=torch.Generator().manual_seed(42)
    )

    train_snapshot_dataset: SnapshotDataset = SnapshotDataset(train_dataset)
    val_snapshot_dataset: SnapshotDataset = SnapshotDataset(val_dataset)
    test_snapshot_dataset: SnapshotDataset = SnapshotDataset(test_dataset)

    train_dataloader: DataLoader = DataLoader(
        dataset=train_snapshot_dataset,
        batch_size=settings.BATCH_SIZE,
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=torch.cuda.is_available(),
    )
    val_dataloader: DataLoader = DataLoader(
        dataset=val_snapshot_dataset,
        batch_size=settings.BATCH_SIZE,
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=torch.cuda.is_available(),
    )
    test_dataloader: DataLoader = DataLoader(
        dataset=test_snapshot_dataset,
        batch_size=settings.BATCH_SIZE,
        shuffle=False,
    )

    return train_dataloader, val_dataloader, test_dataloader