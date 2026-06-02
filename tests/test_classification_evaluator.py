"""Tests for LightningClassificationEvaluator."""

import json
from typing import Any, Dict, List

import pytest
import pytorch_lightning as L
import torch
import torch.nn as nn
from confluid import Class
from dataflux.core import Flux
from dataflux.sample import Sample
from marainer.collate import collate_fn_with_metadata
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection

from sonair.evaluator import LightningClassificationEvaluator


def _make_evaluator(**overrides: Any) -> LightningClassificationEvaluator:
    """Build a LightningClassificationEvaluator, applying overrides as post-construction attributes.

    Mirrors the minimal-ctor pattern: only 'model', 'dataset', 'experiment_name', 'run_name' are
    constructor params (num_classes is read off the model, not configured); deferred slots
    (``loader`` / ``lightning`` / ``metrics`` / ``predictions_sink`` / ``metrics_sidecar_path`` /
    ``class_names``) are set on the instance after construction.
    """
    _ctor_keys = (
        "model",
        "dataset",
        "experiment_name",
        "run_name",
    )
    ctor = {k: overrides.pop(k) for k in _ctor_keys if k in overrides}
    ev = LightningClassificationEvaluator(**ctor)
    for _k, _v in overrides.items():
        setattr(ev, _k, _v)
    return ev


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _TinyClassifier(nn.Module):
    def __init__(self, num_classes: int = 3, input_size: int = 4) -> None:
        super().__init__()
        # Expose num_classes — the evaluator reads the count off the (trained) model.
        self.num_classes = num_classes
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


def test_metrics_are_built_in_initialize() -> None:
    ev = _make_evaluator(
        model=_TinyClassifier(num_classes=3),
        dataset=_make_flux(8),
        num_classes=3,
    )
    # Metrics are the template dict on construction; initialize() builds the collection.
    assert isinstance(ev.metrics, dict)
    ev.initialize()
    assert isinstance(ev.metrics, MetricCollection)


def test_metrics_have_test_prefix() -> None:
    ev = _make_evaluator(
        model=_TinyClassifier(num_classes=3),
        dataset=_make_flux(8),
        num_classes=3,
    )
    ev.initialize()
    keys = list(ev.metrics.keys())  # type: ignore[union-attr]
    assert all(k.startswith("test/") for k in keys), f"Expected test/ prefix, got: {keys}"


def test_default_metrics_have_no_top5() -> None:
    ev = _make_evaluator(
        model=_TinyClassifier(num_classes=10),
        dataset=_make_flux(16, num_classes=10),
        num_classes=10,
    )
    ev.initialize()
    keys = list(ev.metrics.keys())  # type: ignore[union-attr]
    assert not any("top5" in k for k in keys)


def test_top5_opt_in_via_metrics_dict() -> None:
    from confluid import LazyClass
    from torchmetrics.classification import MulticlassAccuracy

    ev = _make_evaluator(
        model=_TinyClassifier(num_classes=10),
        dataset=_make_flux(16, num_classes=10),
        num_classes=10,
        metrics={
            "accuracy_top1": LazyClass(MulticlassAccuracy, top_k=1),
            "accuracy_top5": LazyClass(MulticlassAccuracy, top_k=5),
        },
    )
    ev.initialize()
    keys = list(ev.metrics.keys())  # type: ignore[union-attr]
    assert any("top5" in k for k in keys)


def test_num_classes_read_from_model_not_data() -> None:
    # The evaluator runs a trained model; its class count is authoritative — NOT the eval
    # dataset (which may be a subset). Omit the evaluator num_classes; it comes from the model.
    ev = _make_evaluator(
        model=_TinyClassifier(num_classes=7),
        dataset=_make_flux(8, num_classes=3),  # fewer classes than the model
    )
    ev.initialize()
    assert ev.num_classes == 7  # from the model, not 3 from the data


def test_default_loader_uses_classification_collate_fn() -> None:
    from confluid.fluid import Class as ConfluidClass

    from sonair.classification import classification_collate_fn

    ev = _make_evaluator(
        model=_TinyClassifier(num_classes=3),
        dataset=_make_flux(8),
    )
    assert isinstance(ev.loader, ConfluidClass)
    assert ev.loader.kwargs.get("collate_fn") is classification_collate_fn


# ---------------------------------------------------------------------------
# test_step behaviour
# ---------------------------------------------------------------------------


def test_test_step_updates_metrics() -> None:
    ev = _make_evaluator(
        model=_TinyClassifier(num_classes=3),
        dataset=_make_flux(8),
        num_classes=3,
    )
    ev.initialize()  # build the metric collection from the `metrics` dict (no longer eager)
    batch = Sample(
        input=torch.randn(8, 4),
        target=torch.randint(0, 3, (8,)),
        metadata=[{"idx": i} for i in range(8)],
    )
    ev.test_step(batch, 0)
    computed = ev.metrics.compute()
    assert all(isinstance(v, torch.Tensor) for v in computed.values())


# ---------------------------------------------------------------------------
# Full run with sidecar
# ---------------------------------------------------------------------------


def test_evaluator_run_writes_metrics_sidecar(tmp_path: Any) -> None:
    sidecar = tmp_path / "eval_metrics.json"
    ev = _make_evaluator(
        model=_TinyClassifier(num_classes=3),
        dataset=_make_flux(8),
        num_classes=3,
        loader=Class(
            DataLoader,
            batch_size=4,
            collate_fn=collate_fn_with_metadata,
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


# ---------------------------------------------------------------------------
# String targets: encode via the pinned training map (mirrors the trainer)
# ---------------------------------------------------------------------------


def _string_flux(names: List[str], n: int = 9, input_size: int = 4) -> Flux:
    samples = [
        Sample(input=torch.randn(input_size), target=names[i % len(names)], metadata={"idx": i}) for i in range(n)
    ]
    return Flux(source=samples)


def test_string_targets_encoded_via_checkpoint_class_names(tmp_path: Any) -> None:
    # class_names.json written next to the checkpoint at train time; the evaluator reloads it.
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    (ckpt_dir / "class_names.json").write_text(json.dumps({"class_names": ["bird", "cat", "dog"], "num_classes": 3}))
    model = _TinyClassifier(num_classes=3)
    model.checkpoint_path = str(ckpt_dir / "last.ckpt")  # type: ignore[assignment]

    ev = _make_evaluator(model=model, dataset=_string_flux(["dog", "cat", "bird"]))
    ev.initialize()
    assert ev.num_classes == 3
    assert ev.label_map is not None and ev.label_map.label_names == ["bird", "cat", "dog"]
    # The dataset is now a Flux whose targets are encoded class-id ints.
    encoded = {ev.dataset[i].target for i in range(len(ev.dataset))}
    assert encoded == {0, 1, 2}
    assert all(isinstance(t, int) for t in encoded)


def test_string_targets_encoded_via_explicit_label_map() -> None:
    from dataflux import LabelMap

    lm = LabelMap(mapping={"bird": 0, "cat": 1, "dog": 2})
    ev = _make_evaluator(
        model=_TinyClassifier(num_classes=3),
        dataset=_string_flux(["cat", "dog", "bird"]),
        label_map=lm,
    )
    ev.initialize()
    assert ev.label_map is lm
    assert ev.dataset[0].target == 1  # "cat" -> 1


def test_string_targets_without_any_map_raise() -> None:
    # No label_map and no checkpoint_path → cannot encode string targets.
    ev = _make_evaluator(model=_TinyClassifier(num_classes=3), dataset=_string_flux(["a", "b", "c"]))
    with pytest.raises(RuntimeError):
        ev.initialize()


def test_metrics_accepts_ready_made_collection() -> None:
    # A passed MetricCollection is cloned with the test/ prefix (not left unprefixed); idempotent.
    from torchmetrics.classification import MulticlassAccuracy

    coll = MetricCollection({"top1": MulticlassAccuracy(num_classes=3, top_k=1)})
    ev = _make_evaluator(model=_TinyClassifier(num_classes=3), dataset=_make_flux(8), metrics=coll)
    ev.initialize()
    assert isinstance(ev.metrics, MetricCollection)
    assert all(str(k).startswith("test/") for k in ev.metrics.keys())
    # Idempotent: a second initialize() does not double-prefix.
    ev.initialize()
    assert all(str(k).startswith("test/") and not str(k).startswith("test/test/") for k in ev.metrics.keys())
