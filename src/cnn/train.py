"""Two-stage training loop for ResNet18 transfer learning."""

from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from tqdm.auto import tqdm

from .config import TrainingConfig
from .model import set_backbone_trainable


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    running_loss = 0.0
    predictions: list[int] = []
    targets: list[int] = []
    amp_enabled = scaler is not None and scaler.is_enabled()

    progress = tqdm(loader, leave=False, desc="train" if training else "val")
    for inputs, labels in progress:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(inputs)
                loss = criterion(logits, labels)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        batch_predictions = logits.argmax(dim=1)
        running_loss += loss.item() * labels.size(0)
        predictions.extend(batch_predictions.detach().cpu().tolist())
        targets.extend(labels.detach().cpu().tolist())

    return {
        "loss": running_loss / len(loader.dataset),
        "accuracy": float(np.mean(np.asarray(predictions) == np.asarray(targets))),
        "macro_f1": f1_score(targets, predictions, average="macro", zero_division=0),
    }


def _save_checkpoint(path: Path, model, config: TrainingConfig, history, best_metric, epoch):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "labels": list(config.labels),
            "config": config.serializable_dict(),
            "history": history,
            "best_val_macro_f1": best_metric,
            "best_epoch": epoch,
            "architecture": "resnet18",
            "pretrained_weights": "ResNet18_Weights.DEFAULT (ImageNet-1K)",
        },
        path,
    )


def train_transfer_model(model, loaders, class_weights, config: TrainingConfig, device):
    seed_everything(config.seed)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    scaler = torch.amp.GradScaler(
        device.type, enabled=config.use_amp and device.type == "cuda"
    )
    history: list[dict[str, object]] = []
    best_metric = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    global_epoch = 0
    started = time.perf_counter()

    stages = (
        ("head", config.head_epochs, config.head_learning_rate, False),
        ("finetune", config.finetune_epochs, config.finetune_learning_rate, True),
    )
    for stage, epochs, learning_rate, backbone_trainable in stages:
        set_backbone_trainable(model, backbone_trainable)
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            parameters, lr=learning_rate, weight_decay=config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs)
        )
        epochs_without_improvement = 0

        for stage_epoch in range(1, epochs + 1):
            global_epoch += 1
            train_metrics = _run_epoch(
                model, loaders["train"], criterion, device, optimizer, scaler
            )
            with torch.inference_mode():
                val_metrics = _run_epoch(model, loaders["val"], criterion, device)
            row = {
                "epoch": global_epoch,
                "stage": stage,
                "stage_epoch": stage_epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
            history.append(row)
            print(
                f"epoch={global_epoch:02d} stage={stage:<8} "
                f"train_loss={train_metrics['loss']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} "
                f"val_macro_f1={val_metrics['macro_f1']:.4f}"
            )
            if val_metrics["macro_f1"] > best_metric + 1e-6:
                best_metric = val_metrics["macro_f1"]
                best_epoch = global_epoch
                epochs_without_improvement = 0
                _save_checkpoint(
                    config.best_checkpoint_path,
                    model,
                    config,
                    history,
                    best_metric,
                    best_epoch,
                )
            else:
                epochs_without_improvement += 1
            scheduler.step()
            if epochs_without_improvement >= config.early_stopping_patience:
                print(f"Early stopping stage '{stage}' after epoch {global_epoch}")
                break

    config.output_dir.mkdir(parents=True, exist_ok=True)
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(config.output_dir / "history.csv", index=False)
    elapsed = time.perf_counter() - started
    checkpoint = torch.load(config.best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, history_frame, {"best_epoch": best_epoch, "best_val_macro_f1": best_metric, "seconds": elapsed}

