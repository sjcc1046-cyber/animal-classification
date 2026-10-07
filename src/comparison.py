"""Create a transparent side-by-side summary of the two coursework methods."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


COMMON_METRICS = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]


def build_method_comparison(project_root: Path):
    production_path = project_root / "outputs" / "production_system" / "summary_metrics.csv"
    cnn_path = project_root / "outputs" / "cnn" / "test_metrics.csv"
    if not production_path.exists():
        raise FileNotFoundError("Run 01_production_system.ipynb before comparison")
    if not cnn_path.exists():
        raise FileNotFoundError("Run the CNN experiment before comparison")

    production = pd.read_csv(production_path, index_col=0).loc["总体"]
    cnn = pd.read_csv(cnn_path).iloc[0]
    comparison = pd.DataFrame(
        {
            "Production system (symbolic stress test)": production[COMMON_METRICS],
            "ResNet18 (Animals-10 image test set)": cnn[COMMON_METRICS],
        }
    ).T
    comparison.insert(
        0,
        "Evaluation input",
        ["132 manually generated symbolic cases", "2,265 held-out animal images"],
    )

    output_dir = project_root / "outputs" / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "method_comparison.csv", encoding="utf-8-sig")

    ax = comparison[COMMON_METRICS].T.plot(kind="bar", figsize=(11, 6), ylim=(0, 1.05), rot=0)
    ax.set_title("Method-level metrics (different input test sets)")
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=9, loc="lower right")
    figure = ax.get_figure()
    figure.tight_layout()
    figure.savefig(output_dir / "method_comparison.png", dpi=180, bbox_inches="tight")
    return comparison, figure

