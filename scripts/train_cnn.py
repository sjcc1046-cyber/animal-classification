"""Train and evaluate the six-class ResNet18 model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cnn.config import TrainingConfig
from src.cnn.pipeline import run_experiment


def main() -> None:
    defaults = TrainingConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="train even when a checkpoint exists")
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--workers", type=int, default=defaults.num_workers)
    parser.add_argument("--head-epochs", type=int, default=defaults.head_epochs)
    parser.add_argument("--finetune-epochs", type=int, default=defaults.finetune_epochs)
    args = parser.parse_args()
    config = TrainingConfig(
        batch_size=args.batch_size,
        num_workers=args.workers,
        head_epochs=args.head_epochs,
        finetune_epochs=args.finetune_epochs,
    )
    result = run_experiment(config, force_retrain=args.force)
    print("Test metrics:")
    for name, value in result["summary"].items():
        print(f"  {name}: {value:.4f}" if isinstance(value, float) else f"  {name}: {value}")
    print("Artifacts:", config.output_dir)


if __name__ == "__main__":
    main()
