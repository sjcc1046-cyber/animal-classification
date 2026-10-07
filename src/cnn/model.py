"""ResNet18 construction and freezing helpers."""

from __future__ import annotations

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_resnet18(num_classes: int, pretrained: bool = True, freeze_backbone: bool = True):
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, num_classes)
    set_backbone_trainable(model, not freeze_backbone)
    return model


def set_backbone_trainable(model, trainable: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = trainable
    for parameter in model.fc.parameters():
        parameter.requires_grad = True

