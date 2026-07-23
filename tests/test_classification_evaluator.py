"""Tests for LightningClassifier in the evaluation path (task='test')."""

import json
from typing import Any, Dict, List, Sized, cast

import pytest
import pytorch_lightning as L
import torch
import torch.nn as nn
from confluid import Class
from sampleflux import Flux, Label, Sample
from torchmetrics import MetricCollection

from sonair.classifier import LightningClassifier


def _ts(inp: Any, tgt: Any, **meta: Any) -> Sample:
    """Build a typed-bag Sample: raw `input` field, a Label `class` target, aux meta fields."""
    fields: dict = {"input": inp, "class": Label(tgt)}
    roles: dict = {"class": "target"}
    for _k, _v in meta.items():
        fields[_k] = Label(_v)
        roles[_k] = "aux"
    return Sample(fields, roles)


def _stack_collate(batch: list) -> Sample:
    """A typed-bag DataLoader collate that stacks raw feature/target tensors (replaces the
    legacy marainer.torch.collate_fn_with_metadata, which read the removed Sample triple)."""
    from sampleflux import Label, item_data, primary

    xs = torch.stack([torch.as_tensor(item_data(primary(s, "input")[1])) for s in batch])
    ys = torch.stack([torch.as_tensor(primary(s, "target")[1].value) for s in batch])
    metas: list = []
    for s in batch:
        ik = primary(s, "input")[0]
        tk = primary(s, "target")[0]
        metas.append({k: (v.value if isinstance(v, Label) else v) for k, v in s.items() if k not in (ik, tk)})
    return Sample(
        {"input": xs, "class": Label(value=ys), "metadata": metas}, roles={"class": "target", "metadata": "aux"}
    )


def _make_evaluator(**overrides: Any) -> LightningClassifier:
    """Build a test-path LightningClassifier, applying overrides as post-construction attributes.

    Maps the legacy ``dataset=`` kwarg to ``test_set`` and sets ``task='test'`` (num_classes is read
    off the model). Deferred slots (``test_loader`` / ``lightning`` / ``metrics`` / ``predictions_sink``
    / ``metrics_sidecar_path`` / ``class_names``) are set on the instance after construction.
    """
    ctor: Dict[str, Any] = {"task": "test"}
    for key in ("model", "experiment_name", "run_name", "label_map"):
        if key in overrides:
            ctor[key] = overrides.pop(key)
    if "dataset" in overrides:
        ctor["test_set"] = overrides.pop("dataset")
    overrides.pop("num_classes", None)  # eval reads num_classes off the model
    ev = LightningClassifier(**ctor)
    for _k, _v in overrides.items():
        setattr(ev, _k, _v)
    return ev


class _TinyClassifier(nn.Module):
    def __init__(self, num_classes: int = 3, input_size: int = 4) -> None:
        super().__init__()
        # Expose num_classes — the eval path reads the count off the (trained) model.
        self.num_classes = num_classes
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)  # type: ignore[no-any-return]


def _make_flux(n: int = 8, num_classes: int = 3, input_size: int = 4) -> Flux:
    samples = [_ts(torch.randn(3, 8, 8), torch.tensor(i % num_classes, dtype=torch.int64), idx=i) for i in range(n)]
    return Flux(source=samples)


class _RecordingSink:
    def __init__(self) -> None:
        self.writes: List[Any] = []
        self.closed = False

    def write(self, prediction: Any, metadata: Dict[str, Any]) -> None:
        self.writes.append((prediction, metadata))

    def close(self) -> None:
        self.closed = True


def test_declares_computed_metrics_and_label_names_outputs() -> None:
    from confluid import output_specs

    names = {s["name"] for s in output_specs(LightningClassifier)}
    assert {"computed_metrics", "label_names"} <= names


def test_test_metrics_are_built_in_initialize() -> None:
    ev = _make_evaluator(model=_TinyClassifier(num_classes=3), dataset=_make_flux(8))
    assert isinstance(ev.metrics, dict)  # the template dict on construction
    ev.initialize()
    assert isinstance(ev.test_metrics, MetricCollection)  # built collection


def test_test_metrics_have_test_prefix() -> None:
    ev = _make_evaluator(model=_TinyClassifier(num_classes=3), dataset=_make_flux(8))
    ev.initialize()
    keys = list(ev.test_metrics.keys())
    assert all(str(k).startswith("test/") for k in keys), f"Expected test/ prefix, got: {keys}"


def test_loader_uses_classification_collate_by_default() -> None:
    from sonair.classification import classification_collate_fn

    ev = _make_evaluator(model=_TinyClassifier(), dataset=_make_flux())
    assert ev.test_loader.kwargs["collate_fn"] == classification_collate_fn


def test_run_executes_test_loop(tmp_path: Any) -> None:
    from torch.utils.data import DataLoader

    class _FlattenTinyClassifier(_TinyClassifier):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return super().forward(x.view(x.size(0), -1))

    ev = _make_evaluator(model=_FlattenTinyClassifier(input_size=3 * 8 * 8), dataset=_make_flux())
    # _make_flux yields legacy Samples (the eval initialize() target-walk is legacy-only in this
    # migration stage), so collate them with the legacy marainer collate; the classifier's step
    # functions read the legacy Sample batch through the same _batch_input / _batch_target accessors.
    ev.test_loader = Class(DataLoader, batch_size=8, collate_fn=_stack_collate)
    ev.lightning = Class(
        L.Trainer,
        max_epochs=1,
        limit_test_batches=1,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=str(tmp_path),
    )
    ev.run()
    assert ev.test_metrics is not None


def _string_flux(names: List[str], n: int = 8, input_size: int = 4) -> Flux:
    samples = [_ts(torch.randn(input_size), names[i % len(names)], idx=i) for i in range(n)]
    return Flux(source=samples)


def test_string_targets_encoded_via_checkpoint_class_names(tmp_path: Any) -> None:
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    (ckpt_dir / "class_names.json").write_text(json.dumps({"class_names": ["bird", "cat", "dog"], "num_classes": 3}))
    model = _TinyClassifier(num_classes=3)
    model.checkpoint_path = str(ckpt_dir / "last.ckpt")  # type: ignore[assignment]

    ev = _make_evaluator(model=model, dataset=_string_flux(["dog", "cat", "bird"]))
    ev.initialize()
    assert ev.num_classes == 3
    assert ev.label_map is not None and ev.label_map.label_names == ["bird", "cat", "dog"]
    from sampleflux import iter_targets

    dataset = cast(Flux, ev.test_set)
    assert len(cast(Sized, dataset))  # touch len for parity with the old index-walk
    encoded = set(iter_targets(dataset))
    assert encoded == {0, 1, 2}
    assert all(isinstance(t, int) for t in encoded)


def test_string_targets_encoded_via_explicit_label_map() -> None:
    from sampleflux import LabelMap

    lm = LabelMap(mapping={"bird": 0, "cat": 1, "dog": 2})
    ev = _make_evaluator(
        model=_TinyClassifier(num_classes=3), dataset=_string_flux(["cat", "dog", "bird"]), label_map=lm
    )
    ev.initialize()
    assert ev.label_map is lm
    from sampleflux import iter_targets

    dataset = cast(Flux, ev.test_set)
    assert next(iter(iter_targets(dataset))) == 1  # "cat" -> 1


def test_string_targets_without_any_map_raise() -> None:
    ev = _make_evaluator(model=_TinyClassifier(num_classes=3), dataset=_string_flux(["a", "b", "c"]))
    with pytest.raises(RuntimeError):
        ev.initialize()
