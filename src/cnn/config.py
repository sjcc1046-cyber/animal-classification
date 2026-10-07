"""Configuration for the reproducible ResNet18 transfer-learning experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.labels import LABELS_EN


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class TrainingConfig:
    project_root: Path = field(default_factory=_project_root)
    labels: tuple[str, ...] = LABELS_EN
    image_size: int = 224
    resize_size: int = 256
    batch_size: int = 64
    num_workers: int = 4
    seed: int = 42
    head_epochs: int = 2
    finetune_epochs: int = 6
    head_learning_rate: float = 1e-3
    finetune_learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    early_stopping_patience: int = 3
    use_amp: bool = True

    @property
    def data_root(self) -> Path:
        return self.project_root / "data" / "animals10"

    @property
    def metadata_path(self) -> Path:
        return self.data_root / "metadata.csv"

    @property
    def checkpoint_dir(self) -> Path:
        return self.project_root / "checkpoints" / "cnn"

    @property
    def output_dir(self) -> Path:
        return self.project_root / "outputs" / "cnn"

    @property
    def best_checkpoint_path(self) -> Path:
        return self.checkpoint_dir / "best_resnet18.pt"

    def serializable_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["project_root"] = str(self.project_root)
        values["labels"] = list(self.labels)
        return values

