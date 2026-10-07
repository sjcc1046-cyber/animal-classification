"""End-to-end training, evaluation and artifact generation."""

from __future__ import annotations

import json

import torch

from .config import TrainingConfig
from .dataset import create_dataloaders
from .evaluate import (
    plot_confusion,
    plot_learning_curves,
    predict,
    save_evaluation,
)
from .model import build_resnet18
from .train import seed_everything, train_transfer_model


def run_experiment(config: TrainingConfig, force_retrain: bool = False):
    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, datasets, class_weights = create_dataloaders(
        metadata_path=config.metadata_path,
        data_root=config.data_root,
        labels=config.labels,
        image_size=config.image_size,
        resize_size=config.resize_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
        pin_memory=device.type == "cuda",
    )
    reuse_checkpoint = config.best_checkpoint_path.exists() and not force_retrain
    model = build_resnet18(
        len(config.labels), pretrained=not reuse_checkpoint, freeze_backbone=True
    )
    model.to(device)

    if not reuse_checkpoint:
        model, history, training_info = train_transfer_model(
            model, loaders, class_weights, config, device
        )
    else:
        checkpoint = torch.load(
            config.best_checkpoint_path, map_location=device, weights_only=False
        )
        if tuple(checkpoint["labels"]) != tuple(config.labels):
            raise ValueError("Checkpoint class order differs from current configuration")
        model.load_state_dict(checkpoint["model_state_dict"])
        import pandas as pd

        history = pd.read_csv(config.output_dir / "history.csv")
        previous_seconds = None
        run_info_path = config.output_dir / "run_info.json"
        if run_info_path.exists():
            previous_seconds = json.loads(run_info_path.read_text(encoding="utf-8")).get(
                "seconds"
            )
        training_info = {
            "best_epoch": checkpoint["best_epoch"],
            "best_val_macro_f1": checkpoint["best_val_macro_f1"],
            "seconds": previous_seconds,
        }

    y_true, y_pred, probabilities = predict(model, loaders["test"], device)
    summary, per_class = save_evaluation(
        y_true, y_pred, probabilities, config.labels, config.output_dir
    )
    learning_figure = plot_learning_curves(
        history, config.output_dir / "learning_curves.png"
    )
    confusion_figure = plot_confusion(
        y_true, y_pred, config.labels, config.output_dir / "confusion_matrix.png"
    )
    run_info = {
        "device": str(device),
        "dataset_sizes": {key: len(value) for key, value in datasets.items()},
        **training_info,
    }
    (config.output_dir / "run_info.json").write_text(
        json.dumps(run_info, indent=2), encoding="utf-8"
    )
    return {
        "model": model,
        "history": history,
        "summary": summary,
        "per_class": per_class,
        "run_info": run_info,
        "learning_figure": learning_figure,
        "confusion_figure": confusion_figure,
    }
