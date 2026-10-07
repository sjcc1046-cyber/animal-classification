"""Evaluation tables and plots for the trained CNN."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def predict(model, loader, device):
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    probabilities: list[list[float]] = []
    with torch.inference_mode():
        for inputs, labels in loader:
            logits = model(inputs.to(device, non_blocking=True))
            probs = logits.softmax(dim=1).cpu()
            targets.extend(labels.tolist())
            predictions.extend(probs.argmax(dim=1).tolist())
            probabilities.extend(probs.tolist())
    return np.asarray(targets), np.asarray(predictions), np.asarray(probabilities)


def evaluate_predictions(y_true, y_pred, labels: tuple[str, ...]):
    indexes = list(range(len(labels)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=indexes, zero_division=0
    )
    macro = precision_recall_fscore_support(
        y_true, y_pred, labels=indexes, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        y_true, y_pred, labels=indexes, average="weighted", zero_division=0
    )
    summary = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro Precision": macro[0],
        "Macro Recall": macro[1],
        "Macro F1": macro[2],
        "Weighted F1": weighted[2],
        "Sample count": len(y_true),
    }
    per_class = pd.DataFrame(
        {"Precision": precision, "Recall": recall, "F1": f1, "Support": support},
        index=labels,
    )
    per_class.index.name = "Class"
    return summary, per_class


def save_evaluation(y_true, y_pred, probabilities, labels, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    summary, per_class = evaluate_predictions(y_true, y_pred, labels)
    pd.DataFrame([summary]).to_csv(output_dir / "test_metrics.csv", index=False)
    per_class.to_csv(output_dir / "per_class_metrics.csv")
    prediction_frame = pd.DataFrame(
        {
            "true_index": y_true,
            "predicted_index": y_pred,
            "true_label": [labels[index] for index in y_true],
            "predicted_label": [labels[index] for index in y_pred],
            "confidence": probabilities.max(axis=1),
        }
    )
    for index, label in enumerate(labels):
        prediction_frame[f"prob_{label}"] = probabilities[:, index]
    prediction_frame.to_csv(output_dir / "test_predictions.csv", index=False)
    (output_dir / "test_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary, per_class


def plot_learning_curves(history: pd.DataFrame, output_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for axis, metric, title in zip(
        axes,
        ("loss", "accuracy", "macro_f1"),
        ("Loss", "Accuracy", "Macro F1"),
    ):
        axis.plot(history["epoch"], history[f"train_{metric}"], marker="o", label="train")
        axis.plot(history["epoch"], history[f"val_{metric}"], marker="o", label="validation")
        axis.axvline(history.loc[history["stage"] == "finetune", "epoch"].min(), color="gray", linestyle="--", alpha=0.6)
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("ResNet18 transfer-learning history")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig


def plot_confusion(y_true, y_pred, labels, output_path: Path):
    counts = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    normalized = counts / counts.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    for row in range(len(labels)):
        for column in range(len(labels)):
            value = normalized[row, column]
            ax.text(column, row, f"{value:.2f}\n({counts[row, column]})", ha="center", va="center", color="white" if value > 0.5 else "black", fontsize=8)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("CNN row-normalized confusion matrix")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig

