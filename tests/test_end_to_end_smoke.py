"""End-to-end smoke test: TimmClassifierModel + torch.nn.CrossEntropyLoss +
classification_collate_fn + a fake HuggingFaceClassificationDataset
threaded through one full forward+backward step.

This proves the wiring contract that the YAML configs in
``waivefront-rfuav/config/`` exercise — the actual RFUAV training flow
just adds the HF Hub fetch in front. Useful as a CI gate so the
classification path stays green even when HuggingFace's CDN is rate-limited
or unreachable.
"""

from typing import List
from unittest.mock import patch

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

from sonair.classification import classification_collate_fn
from sonair.hf_dataset import HuggingFaceClassificationDataset
from sonair.timm_classifier import TimmClassifierForPrediction, TimmClassifierModel


class _FakeFeature:
    def __init__(self, names: List[str]) -> None:
        self.names = list(names)


class _FakeHFDataset:
    def __init__(self, n_per_class: int = 4, n_classes: int = 3, image_size: int = 64) -> None:
        self._items = []
        for cls_id in range(n_classes):
            for _ in range(n_per_class):
                # Each class gets a distinct base color so the model has
                # some signal to learn (not strictly necessary for a smoke
                # test, but makes the loss meaningful).
                color = ((cls_id * 80) % 255, (cls_id * 50) % 255, (cls_id * 30) % 255)
                self._items.append(
                    {
                        "image": Image.new("RGB", (image_size, image_size), color=color),
                        "label": cls_id,
                    }
                )
        self.features = {"label": _FakeFeature([f"class_{i}" for i in range(n_classes)])}

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict:
        return self._items[idx]


def test_full_train_step_runs_end_to_end() -> None:
    """One full training iteration: dataset → loader → model → loss → backward."""
    fake = _FakeHFDataset(n_per_class=4, n_classes=3, image_size=64)
    with patch("datasets.load_dataset", return_value=fake):
        ds = HuggingFaceClassificationDataset(
            repo_id="fake/RFUAV",
            split="train",
            image_size=64,
        )

    loader = DataLoader(
        ds,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=classification_collate_fn,
    )
    # resnet18 is small and runs in a few hundred ms; use 64x64 input so
    # this whole test stays under one second wall-clock.
    model = TimmClassifierModel(
        model_name="resnet18",
        num_classes=3,
        pretrained=False,
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    batch = next(iter(loader))
    assert batch.input.shape == (4, 3, 64, 64)
    assert batch.target.shape == (4,)
    assert batch.target.dtype == torch.int64

    logits = model(batch.input)
    assert logits.shape == (4, 3)

    loss = loss_fn(logits, batch.target)
    assert loss.dim() == 0
    assert torch.isfinite(loss)

    optimizer.zero_grad()
    loss.backward()
    # At least one parameter accumulated a non-zero grad.
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    optimizer.step()


def test_prediction_class_emits_per_sample_dicts_for_full_batch() -> None:
    """The predict YAML's path: TimmClassifierForPrediction yields a list of dicts."""
    fake = _FakeHFDataset(n_per_class=2, n_classes=3, image_size=64)
    with patch("datasets.load_dataset", return_value=fake):
        ds = HuggingFaceClassificationDataset(repo_id="fake/RFUAV", split="train", image_size=64)
    loader = DataLoader(ds, batch_size=4, num_workers=0, collate_fn=classification_collate_fn)

    model = TimmClassifierForPrediction(
        model_name="resnet18",
        num_classes=3,
        pretrained=False,
    ).eval()

    batch = next(iter(loader))
    outputs = model(batch.input)
    assert isinstance(outputs, list)
    assert len(outputs) == batch.input.shape[0]
    for out in outputs:
        assert set(out.keys()) == {"logits", "probs", "class_idx"}
        assert out["logits"].shape == (1, 3)
        assert torch.isfinite(out["logits"]).all()


def test_metadata_per_sample_survives_collation() -> None:
    """Per-sample metadata (label_name, repo_id, etc.) must round-trip through collate."""
    fake = _FakeHFDataset(n_per_class=2, n_classes=3, image_size=32)
    with patch("datasets.load_dataset", return_value=fake):
        ds = HuggingFaceClassificationDataset(repo_id="fake/RFUAV", split="train", image_size=32)
    loader = DataLoader(ds, batch_size=4, num_workers=0, collate_fn=classification_collate_fn)
    batch = next(iter(loader))
    metas = batch.metadata["per_sample"]
    assert isinstance(metas, list)
    assert len(metas) == 4
    for m in metas:
        assert m["repo_id"] == "fake/RFUAV"
        assert m["split"] == "train"
        assert m["label_name"].startswith("class_")
