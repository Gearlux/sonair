"""End-to-end smoke test: TimmModel + torch.nn.CrossEntropyLoss +
``recordstream.collate_records`` + a fake record dataset threaded through one full
forward+backward step.

This proves the wiring contract that the YAML configs in
``waivefront/config/`` exercise — the actual RFUAV training flow
just adds the HF Hub fetch in front. The fake dataset yields the tensorized
record dict that ``HuggingFaceSource`` produces (a ``[3, H, W]`` float
``Image`` under ``"image"`` + a ``Label`` under ``"class"`` + plain metadata
keys), collated by the generic ``collate_records`` and read back through the
same ``_batch_input`` / ``_batch_target`` accessors the classifier uses. Useful
as a CI gate so the classification path stays green even when HuggingFace's CDN
is rate-limited.
"""

from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from recordstream import Image, Label, Record, collate_records
from torch.utils.data import DataLoader

from sonair.lightning_classifier import _batch_input, _batch_metadata, _batch_target
from sonair.timm_classifier import TimmModel


class _FakeFeature:
    def __init__(self, names: List[str]) -> None:
        self.names = list(names)


class _FakeHFDataset:
    """Fake HF dataset that returns record dicts (same interface as HuggingFaceSource)."""

    def __init__(self, n_per_class: int = 4, n_classes: int = 3, image_size: int = 64) -> None:
        self._items: List[Dict[str, Any]] = []
        self.image_size = image_size
        for cls_id in range(n_classes):
            for _ in range(n_per_class):
                # Each class gets a distinct base color so the model has some signal to learn.
                color = ((cls_id * 80) % 255, (cls_id * 50) % 255, (cls_id * 30) % 255)
                image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
                image[:, :] = color
                self._items.append({"image": image, "label": cls_id})
        self.features = {"label": _FakeFeature([f"class_{i}" for i in range(n_classes)])}

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Record:
        item = self._items[idx]
        label = item["label"]
        # The pipeline (ConvertToImage → ToTensor) has already produced a [3, H, W] float image, matching
        # HuggingFaceSource's post-pipeline output: a CHW float Image, an int-id target Label, plain meta keys.
        chw = np.transpose(item["image"], (2, 0, 1)).astype("float32") / 255.0
        return {
            "image": Image(chw, layout="CHW"),
            "class": Label(label),
            "label_idx": label,
            "label_name": f"class_{label}",
        }


def test_full_train_step_runs_end_to_end() -> None:
    """One full training iteration: dataset → loader → record collate → model → loss → backward."""
    fake_ds = _FakeHFDataset(n_per_class=4, n_classes=3, image_size=64)

    # recordstream sources are map-style (len + getitem) but not torch Dataset
    # subclasses, so the stub's arg-type is stricter than DataLoader's runtime.
    loader: DataLoader = DataLoader(
        fake_ds,  # type: ignore[arg-type]
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_records,
    )
    # resnet18 is small and runs in a few hundred ms; use 64x64 input so
    # this whole test stays under one second wall-clock.
    model = TimmModel(
        model_name="resnet18",
        num_classes=3,
        pretrained=False,
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Solidify the lazy model so optimizer can see parameters.
    model.solidify()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    batch = next(iter(loader))
    x, y = _batch_input(batch, "image"), _batch_target(batch, "class")
    assert x.shape == (4, 3, 64, 64)
    assert y.shape == (4,)
    assert y.dtype == torch.int64

    logits = model(x)
    assert logits.shape == (4, 3)

    loss = loss_fn(logits, y)
    assert loss.dim() == 0
    assert torch.isfinite(loss)

    optimizer.zero_grad()
    loss.backward()
    # At least one parameter accumulated a non-zero grad.
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    optimizer.step()


def test_metadata_per_record_survives_collation() -> None:
    """Per-record metadata must round-trip through the record collate as a list of dicts."""
    fake_ds = _FakeHFDataset(n_per_class=2, n_classes=3, image_size=32)
    loader: DataLoader = DataLoader(
        fake_ds,  # type: ignore[arg-type]
        batch_size=4,
        num_workers=0,
        collate_fn=collate_records,
    )
    batch = next(iter(loader))
    metas = _batch_metadata(batch, "image", "class")
    assert isinstance(metas, list)
    assert len(metas) == 4
    for m in metas:
        assert m["label_name"].startswith("class_")
