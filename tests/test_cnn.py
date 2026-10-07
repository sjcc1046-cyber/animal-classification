import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from src.cnn.dataset import AnimalImageDataset, read_metadata
from src.cnn.model import build_resnet18, set_backbone_trainable
from src.labels import LABELS_EN


class CnnComponentTests(unittest.TestCase):
    def test_model_output_and_freezing(self):
        model = build_resnet18(len(LABELS_EN), pretrained=False, freeze_backbone=True)
        self.assertTrue(all(not p.requires_grad for name, p in model.named_parameters() if not name.startswith("fc.")))
        self.assertTrue(all(p.requires_grad for p in model.fc.parameters()))
        output = model(torch.randn(2, 3, 64, 64))
        self.assertEqual(tuple(output.shape), (2, len(LABELS_EN)))
        torch.nn.functional.cross_entropy(output, torch.tensor([0, 1])).backward()
        self.assertIsNotNone(model.fc.weight.grad)
        reloaded = build_resnet18(len(LABELS_EN), pretrained=False, freeze_backbone=True)
        reloaded.load_state_dict(model.state_dict())
        self.assertTrue(torch.equal(reloaded.fc.weight, model.fc.weight))
        set_backbone_trainable(model, True)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_metadata_dataset_and_split_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for split in ("train", "val", "test"):
                for label in LABELS_EN:
                    path = Path("images") / label / f"{split}.jpg"
                    absolute = root / path
                    absolute.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (16, 16), color="white").save(absolute)
                    rows.append({"path": path.as_posix(), "label_en": label, "split": split})
            metadata_path = root / "metadata.csv"
            pd.DataFrame(rows).to_csv(metadata_path, index=False)
            frame = read_metadata(metadata_path)
            dataset = AnimalImageDataset(
                frame[frame["split"] == "train"], root, transform=transforms.ToTensor()
            )
            image, target = dataset[0]
            self.assertEqual(tuple(image.shape), (3, 16, 16))
            self.assertEqual(target, 0)
            self.assertEqual(len(dataset), len(LABELS_EN))


if __name__ == "__main__":
    unittest.main()
