"""Download and prepare six selected classes from Rapidata/Animals-10.

The source Parquet file is cached by Hugging Face. Images are exported once to
``data/animals10/images/<english_label>`` and a deterministic stratified split is
written to ``data/animals10/metadata.csv``. No image is duplicated across splits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from PIL import Image


DATASET_ID = "Rapidata/Animals-10"
DATA_FILE = "hf://datasets/Rapidata/Animals-10/data/train-00000-of-00001.parquet"
SELECTED_LABELS = ("Cat", "Dog", "Chicken", "Horse", "Cow", "Sheep")
LABEL_ZH = {
    "Cat": "猫",
    "Dog": "狗",
    "Chicken": "鸡",
    "Horse": "马",
    "Cow": "牛",
    "Sheep": "羊",
}


def stable_split(label: str, source_index: int) -> str:
    """Return a reproducible approximately 70/15/15 split."""
    digest = hashlib.sha256(f"animals10:{label}:{source_index}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "val"
    return "test"


def _to_rgb_image(value) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict) and value.get("bytes") is not None:
        return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
    if isinstance(value, dict) and value.get("path"):
        return Image.open(value["path"]).convert("RGB")
    raise TypeError(f"Unsupported image value: {type(value)!r}")


def prepare(output: Path, force: bool = False) -> None:
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output / "metadata.csv"
    if metadata_path.exists() and not force:
        raise FileExistsError(
            f"{metadata_path} already exists; use --force to verify and overwrite exports"
        )

    print(f"Loading {DATASET_ID} ...")
    dataset = load_dataset("parquet", data_files={"train": DATA_FILE}, split="train")
    label_feature = dataset.features["label"]
    label_names = tuple(label_feature.names)
    wanted = set(SELECTED_LABELS)
    missing = wanted - set(label_names)
    if missing:
        raise RuntimeError(f"Dataset is missing labels: {sorted(missing)}")

    rows: list[dict[str, str | int]] = []
    counts: Counter[tuple[str, str]] = Counter()
    for source_index, sample in enumerate(dataset):
        label_en = label_names[sample["label"]]
        if label_en not in wanted:
            continue
        class_dir = output / "images" / label_en.lower()
        class_dir.mkdir(parents=True, exist_ok=True)
        relative_path = Path("images") / label_en.lower() / f"{source_index:06d}.jpg"
        image_path = output / relative_path
        if force or not image_path.exists():
            with _to_rgb_image(sample["image"]) as image:
                image.save(image_path, format="JPEG", quality=95)
        split = stable_split(label_en, source_index)
        counts[(label_en, split)] += 1
        rows.append(
            {
                "path": relative_path.as_posix(),
                "label_en": label_en.lower(),
                "label_zh": LABEL_ZH[label_en],
                "split": split,
                "source_index": source_index,
            }
        )

    with metadata_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("path", "label_en", "label_zh", "split", "source_index")
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "source": DATASET_ID,
        "license": "GPL-2.0 (as stated by the dataset card)",
        "selected_labels": {key.lower(): value for key, value in LABEL_ZH.items()},
        "split_method": "SHA-256(label, source_index), approximately 70/15/15",
        "counts": {
            label.lower(): {split: counts[(label, split)] for split in ("train", "val", "test")}
            for label in SELECTED_LABELS
        },
        "total": len(rows),
    }
    (output / "dataset_info.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/animals10"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    prepare(args.output.resolve(), args.force)


if __name__ == "__main__":
    main()
