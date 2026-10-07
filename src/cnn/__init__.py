"""ResNet18 transfer-learning components for six-class image recognition."""

from .config import TrainingConfig
from .dataset import AnimalImageDataset, create_dataloaders
from .model import build_resnet18, set_backbone_trainable

__all__ = [
    "AnimalImageDataset",
    "TrainingConfig",
    "build_resnet18",
    "create_dataloaders",
    "set_backbone_trainable",
]

