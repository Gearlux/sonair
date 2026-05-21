"""Tests for LightningClassificationEvaluator."""

import json
from typing import Any, Dict, List

import pytorch_lightning as L
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection

from confluid import Class
from dataflux.core import Flux
from dataflux.sample import Sample
from sonair.classification import classification_collate_fn
from sonair.evaluator import LightningClassificationEvaluator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _TinyClassifier(nn.Module):
    def __init__(self, num_classes: int = 3, input_size: int = 4) -> None:
        super().__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)  # type: ignore[no-any-return]


def _make_flux(n: int = 8, num_classes: int = 3, input_size: int = 4) -> Flux:
    samples = [
        Sample(
            input=torch.randn(input_size),
            target=torch.tensor(i % num_classes, dtype=torch.int64),
            metadata={"idx": i},
        )
        for i in range(n)
    ]
    return Flux(source=samples)


class _RecordingSink:
    def __init__(self) -> None:
        self.writes: List[Any] = []
        self.closed = False

    def write(self, prediction: Any, metadata: Dict[str, Any]) -> None:
        self.writes.append((prediction, metadata))

    def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# LightningClassificationEvaluator defaults
# ---------------------------------------------------------------------------


def test_metrics_are_wired_on_construction() -> None:
    ev = LightningClassificationEvaluator(
        model=_TinyClassifier(num_classes=3),
        dataset=_make_flux(8),
        num_classes=3,
    )
    assert ev.metrics is not None
    assert isinstance(ev.metrics, MetricCollection)


def test_metrics_have_test_prefix() -> None:
    ev = LightningClassificationEvaluator(
        model=_TinyClassifier(num_classes=3),
        dataset=_make_flux(8),
        num_classes=3,
    )
    keys = list(ev.metrics.keys())  # type: ignore[union-attr]
    assert all(
        k.startswith("test/") for k in keys
    ), f"Expected test/ prefix, got: {keys}"


def test_no_top5_for_small_num_classes() -> None:
    ev = LightningClassificationEvaluator(
        model=_TinyClassifier(num_classes=2),
        dataset=_make_flux(8, num_classes=2),
        num_classes=2,
    )
    assert ev.metrics is not None
    keys = list(ev.metrics.keys())
    assert not any("top5" in k for k in keys)


def test_top5_present_for_large_num_classes() -> None:
    ev = LightningClassificationEvaluator(
        model=_TinyClassifier(num_classes=10),
        dataset=_make_flux(16, num_classes=10),
        num_classes=10,
    )
    assert ev.metrics is not None
    keys = list(ev.metrics.keys())
    assert any("top5" in k for k in keys)


def test_default_loader_uses_classification_collate_fn() -> None:
    from confluid.fluid import Class as ConfluidClass

    ev = LightningClassificationEvaluator(
        model=_TinyClassifier(num_classes=3),
        dataset=_make_flux(8),
        num_classes=3,
    )
    assert isinstance(ev.loader, ConfluidClass)
    assert ev.loader.kwargs.get("collate_fn") is classification_collate_fn


# ---------------------------------------------------------------------------
# test_step behaviour
# ---------------------------------------------------------------------------


def test_test_step_updates_metrics() -> None:
    ev = LightningClassificationEvaluator(
        model=_TinyClassifier(num_classes=3),
        dataset=_make_flux(8),
        num_classes=3,
    )
    batch = Sample(
        input=torch.randn(8, 4),
        target=torch.randint(0, 3, (8,)),
        metadata={"per_sample": [{"idx": i} for i in range(8)]},
    )
    ev.test_step(batch, 0)
    assert ev.metrics is not None
    computed = ev.metrics.compute()
    assert all(isinstance(v, torch.Tensor) for v in computed.values())


# ---------------------------------------------------------------------------
# Full run with sidecar
# ---------------------------------------------------------------------------


def test_evaluator_run_writes_metrics_sidecar(tmp_path: Any) -> None:
    sidecar = tmp_path / "eval_metrics.json"
    ev = LightningClassificationEvaluator(
        model=_TinyClassifier(num_classes=3),
        dataset=_make_flux(8),
        num_classes=3,
        loader=Class(
            DataLoader,
            batch_size=4,
            collate_fn=classification_collate_fn,
        ),
        lightning=Class(
            L.Trainer,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            default_root_dir=str(tmp_path),
        ),
        experiment_name="test_exp",
        run_name="test_run",
        metrics_sidecar_path=str(sidecar),
    )
    ev.run()
    assert sidecar.exists(), "Sidecar must be written."
    payload = json.loads(sidecar.read_text())
    assert payload["experiment_name"] == "test_exp"
    assert "metrics" in payload
    metric_values = payload["metrics"]
    assert any("accuracy" in k or "f1" in k for k in metric_values)
    for v in metric_values.values():
        assert isinstance(v, (int, float))
