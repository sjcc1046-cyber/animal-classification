"""Metadata-driven datasets and loaders for the six Animals-10 classes."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights

from src.labels import LABELS_EN


class AnimalImageDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        data_root: Path,
        labels: tuple[str, ...] = LABELS_EN,
        transform: Callable | None = None,
    ) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.data_root = Path(data_root)
        self.transform = transform
        self.labels = tuple(labels)
        self.class_to_index = {name: index for index, name in enumerate(self.labels)}
        unknown = set(self.frame["label_en"]) - set(self.labels)
        if unknown:
            raise ValueError(f"Unknown labels in metadata: {sorted(unknown)}")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.frame.iloc[index]
        path = self.data_root / row["path"]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, self.class_to_index[row["label_en"]]


def build_transforms(image_size: int = 224, resize_size: int = 256):
    weights = ResNet18_Weights.DEFAULT
    mean = weights.transforms().mean
    std = weights.transforms().std
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.70, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_transform, eval_transform


def read_metadata(metadata_path: Path, labels: tuple[str, ...] = LABELS_EN) -> pd.DataFrame:
    frame = pd.read_csv(metadata_path, encoding="utf-8-sig")
    required = {"path", "label_en", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Metadata is missing columns: {sorted(missing)}")
    if frame["path"].duplicated().any():
        raise ValueError("The same image path appears more than once in metadata")
    if set(frame["split"]) != {"train", "val", "test"}:
        raise ValueError("Metadata must contain train, val and test splits")
    if set(frame["label_en"]) != set(labels):
        raise ValueError("Metadata labels do not match the configured six classes")
    return frame


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloaders(
    metadata_path: Path,
    data_root: Path,
    labels: tuple[str, ...] = LABELS_EN,
    image_size: int = 224,
    resize_size: int = 256,
    batch_size: int = 64,
    num_workers: int = 4,
    seed: int = 42,
    pin_memory: bool = True,
):
    frame = read_metadata(metadata_path, labels)
    train_transform, eval_transform = build_transforms(image_size, resize_size)
    datasets = {
        split: AnimalImageDataset(
            frame[frame["split"] == split],
            data_root,
            labels,
            train_transform if split == "train" else eval_transform,
        )
        for split in ("train", "val", "test")
    }
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            worker_init_fn=_seed_worker,
            generator=generator,
        )
        for split, dataset in datasets.items()
    }
    train_counts = (
        frame[frame["split"] == "train"]["label_en"]
        .value_counts()
        .reindex(labels)
        .to_numpy(dtype=np.float32)
    )
    class_weights = train_counts.sum() / (len(labels) * train_counts)
    return loaders, datasets, torch.tensor(class_weights, dtype=torch.float32)

