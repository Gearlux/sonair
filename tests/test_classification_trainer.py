"""Tests for LightningClassifier."""

from typing import Any, cast

import pytest
import torch
import torch.nn as nn
from confluid import LazyClass
from sampleflux import Flux
from sampleflux.sample import Sample
from torchmetrics import MetricCollection

from sonair.classifier import LightningClassifier

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _TinyClassifier(nn.Module):
    def __init__(self, num_classes: int = 3, input_size: int = 4) -> None:
        super().__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)  # type: ignore[no-any-return]


def _make_batch(n: int = 8, num_classes: int = 3, input_size: int = 4) -> Sample:
    return Sample(
        input=torch.randn(n, input_size),
        target=torch.randint(0, num_classes, (n,)),
        metadata=[{"idx": i} for i in range(n)],
    )


def _make_samples(n: int = 16, num_classes: int = 3, input_size: int = 4) -> Flux:
    # A Flux (torch Dataset) — the trainer's train_set is typed Dataset[Any].
    return Flux(source=[Sample(input=torch.randn(input_size), target=i % num_classes, metadata={}) for i in range(n)])


# ---------------------------------------------------------------------------
# LightningClassifier defaults
# ---------------------------------------------------------------------------


def test_loss_fn_defaults_to_cross_entropy() -> None:
    trainer = LightningClassifier(
        model=_TinyClassifier(num_classes=3),
        train_set=_make_samples(16),
        num_classes=3,
    )
    assert isinstance(trainer.loss_fn, nn.CrossEntropyLoss)


def test_initialize_wires_metrics_with_correct_prefixes() -> None:
    trainer = LightningClassifier(
        model=_TinyClassifier(num_classes=3),
        train_set=_make_samples(16),
        num_classes=3,
    )
    # Metrics are built in initialize() (not __init__) from the `metrics` dict.
    assert trainer.train_metrics is None
    trainer.initialize()
    assert isinstance(trainer.train_metrics, MetricCollection)
    assert isinstance(trainer.val_metrics, MetricCollection)
    assert isinstance(trainer.test_metrics, MetricCollection)

    train_keys = [str(k) for k in trainer.train_metrics.keys()]
    val_keys = [str(k) for k in trainer.val_metrics.keys()]
    test_keys = [str(k) for k in trainer.test_metrics.keys()]

    assert all(k.startswith("train/") for k in train_keys), f"Expected train/ prefix, got: {train_keys}"
    assert all(k.startswith("val/") for k in val_keys), f"Expected val/ prefix, got: {val_keys}"
    assert all(k.startswith("test/") for k in test_keys), f"Expected test/ prefix, got: {test_keys}"


def test_default_metrics_have_no_top5() -> None:
    # top-5 is no longer auto-added; the default metrics dict is top-1 + macro F1.
    trainer = LightningClassifier(
        model=_TinyClassifier(num_classes=10),
        train_set=_make_samples(16, num_classes=10),
        num_classes=10,
    )
    trainer.initialize()
    assert trainer.train_metrics is not None
    keys = [str(k) for k in trainer.train_metrics.keys()]
    assert not any("top5" in k for k in keys), f"top5 should be opt-in, got: {keys}"


def test_top5_opt_in_via_metrics_dict() -> None:
    # top-5 is opt-in: add it to the `metrics` dict; initialize() flows it with num_classes.
    from confluid import LazyClass
    from torchmetrics.classification import MulticlassAccuracy

    trainer = LightningClassifier(
        model=_TinyClassifier(num_classes=10),
        train_set=_make_samples(16, num_classes=10),
        num_classes=10,
        metrics={
            "accuracy_top1": LazyClass(MulticlassAccuracy, top_k=1),
            "accuracy_top5": LazyClass(MulticlassAccuracy, top_k=5),
        },
    )
    trainer.initialize()
    assert trainer.train_metrics is not None
    keys = [str(k) for k in trainer.train_metrics.keys()]
    assert any("top5" in k for k in keys), f"top5 should appear when added to metrics dict, got: {keys}"


# ---------------------------------------------------------------------------
# Forward / step methods
# ---------------------------------------------------------------------------


def test_training_step_returns_scalar_loss() -> None:
    trainer = LightningClassifier(
        model=_TinyClassifier(num_classes=3),
        train_set=_make_samples(16),
        num_classes=3,
    )
    batch = _make_batch(n=8, num_classes=3)
    loss = trainer.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_validation_step_returns_scalar_loss() -> None:
    trainer = LightningClassifier(
        model=_TinyClassifier(num_classes=3),
        train_set=_make_samples(16),
        num_classes=3,
    )
    batch = _make_batch(n=8, num_classes=3)
    loss = trainer.validation_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0


def test_metrics_update_on_training_step() -> None:
    trainer = LightningClassifier(
        model=_TinyClassifier(num_classes=3),
        train_set=_make_samples(16),
        num_classes=3,
    )
    trainer.initialize()  # build metrics from the `metrics` dict (no longer eager in __init__)
    batch = _make_batch(n=8, num_classes=3)
    trainer.training_step(batch, 0)
    assert trainer.train_metrics is not None
    computed = trainer.train_metrics.compute()
    assert all(isinstance(v, torch.Tensor) for v in computed.values())


# ---------------------------------------------------------------------------
# Default loader collate is baked in
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Omitted num_classes -> deferred metrics, derived from the dataset at run time
# ---------------------------------------------------------------------------


class _DimsDataset(torch.utils.data.Dataset):
    """Map-style torch Dataset of integer-target samples."""

    def __init__(self, n: int, num_classes: int, input_size: int = 4) -> None:
        self._s = [
            Sample(input=torch.randn(input_size), target=torch.tensor(i % num_classes, dtype=torch.int64), metadata={})
            for i in range(n)
        ]
        self._nc = num_classes

    def __len__(self) -> int:
        return len(self._s)

    def __getitem__(self, i: int) -> Sample:
        return self._s[i]

    def derived_dimensions(self) -> dict:
        return {"num_classes": self._nc}


def test_omitted_num_classes_leaves_metrics_none_until_run() -> None:
    """With num_classes omitted, metrics stay None post-construction — sonair builds them
    in run() once it derives the count (see test_omitted_num_classes_derived_and_built_at_run),
    rather than carrying a deferred Lazy fluid."""
    trainer = LightningClassifier(
        model=_TinyClassifier(num_classes=4),
        train_set=_make_samples(16, num_classes=4),
    )
    assert trainer.num_classes is None
    assert trainer.train_metrics is None
    assert trainer.val_metrics is None
    assert trainer.test_metrics is None


def _cpu_lightning(tmp_path: object) -> Any:
    """A test-friendly Lightning Trainer Class stub (cpu, 1 batch, no logger)."""
    import pytorch_lightning as L
    from confluid import Class as ConfluidClass

    return ConfluidClass(
        L.Trainer,
        max_epochs=1,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        limit_train_batches=1,
        default_root_dir=str(tmp_path),
    )


def test_omitted_num_classes_derived_and_built_at_run(tmp_path: object) -> None:
    from confluid import Class as ConfluidClass
    from marainer.torch import collate_fn_with_metadata
    from torch.utils.data import DataLoader

    # Minimal constructor; everything else configured post-construction (the paradigm).
    trainer = LightningClassifier(
        model=_TinyClassifier(num_classes=4),
        train_set=_DimsDataset(16, num_classes=4),
        # num_classes omitted -> derived from the dataset
    )
    trainer.train_loader = ConfluidClass(DataLoader, batch_size=8, collate_fn=collate_fn_with_metadata)
    trainer.lightning = _cpu_lightning(tmp_path)
    trainer.run()
    assert isinstance(trainer.val_metrics, MetricCollection)
    sized = [m for m in trainer.val_metrics.values() if hasattr(m, "num_classes")]
    assert sized and all(m.num_classes == 4 for m in sized)


def test_default_loader_uses_classification_collate_fn() -> None:
    from confluid.fluid import Class as ConfluidClass

    from sonair.classification import classification_collate_fn

    trainer = LightningClassifier(
        model=_TinyClassifier(num_classes=3),
        train_set=_make_samples(16),
        num_classes=3,
    )
    # The train_loader is a deferred Class stub; check its kwargs contain the collate_fn.
    assert isinstance(trainer.train_loader, ConfluidClass)
    assert trainer.train_loader.kwargs.get("collate_fn") is classification_collate_fn


# ---------------------------------------------------------------------------
# Ergonomic knobs (batch_size / num_workers / persist_workers / max_epochs) are
# defaulted attributes folded into the deferred loader + Lightning stubs at run().
# ---------------------------------------------------------------------------


def test_ergonomic_knobs_reach_loader_and_lightning_stubs() -> None:
    from confluid import Class as ConfluidClass

    # batch_size / num_workers are ctor knobs baked into the deferred loader stub in
    # __init__ (the constructor stays side-effect-free — LazyClass defers construction).
    trainer = LightningClassifier(
        model=_TinyClassifier(num_classes=3),
        train_set=_make_samples(16),
        num_classes=3,
        batch_size=64,
        num_workers=4,
        max_epochs=7,
    )
    tl = trainer.train_loader.kwargs
    assert tl["batch_size"] == 64 and tl["num_workers"] == 4
    assert tl["shuffle"] is True and tl["persistent_workers"] is True
    assert trainer.val_loader.kwargs["shuffle"] is False
    # max_epochs is baked into the base's deferred L.Trainer stub at construction.
    assert isinstance(trainer.lightning, ConfluidClass)
    assert trainer.lightning.kwargs["max_epochs"] == 7


def test_persistent_workers_guarded_when_num_workers_zero() -> None:
    # persistent_workers=True is meaningless (and raises in torch) when num_workers=0,
    # so __init__ derives it as `num_workers != 0`.
    trainer = LightningClassifier(
        model=_TinyClassifier(num_classes=3),
        train_set=_make_samples(16),
        num_classes=3,
        num_workers=0,
    )
    assert trainer.train_loader.kwargs["persistent_workers"] is False


def test_explicit_loader_override_is_respected() -> None:
    from confluid import Class as ConfluidClass
    from torch.utils.data import DataLoader

    # __init__ bakes the ctor batch_size into the default loader stub...
    trainer = LightningClassifier(
        model=_TinyClassifier(num_classes=3),
        train_set=_make_samples(16),
        num_classes=3,
        batch_size=8,
    )
    assert trainer.train_loader.kwargs["batch_size"] == 8
    # ...but a post-construction explicit loader replaces it wholesale (nothing folds
    # over it at run() — the base run() flows whatever loader is wired).
    trainer.train_loader = ConfluidClass(DataLoader, batch_size=999)
    assert trainer.train_loader.kwargs["batch_size"] == 999


# ---------------------------------------------------------------------------
# Flat-config broadcasting + logger deferred-construction (the user's ask)
# ---------------------------------------------------------------------------


_FLAT_YAML = """
trainer: !class:sonair.LightningClassifier
  run_name: x
model: "MODEL_SENTINEL"
train_set: !class:sampleflux.Flux()
val_set: "VAL_SENTINEL"
batch_size: 64
num_workers: 4
max_epochs: 7
logger:
  - !class:pytorch_lightning.loggers.TensorBoardLogger
    save_dir: ./_sonair_test_logs
    name: bcast
"""


def test_flat_broadcasting_wires_trainer_attributes_from_top_level_keys() -> None:
    """Top-level sibling keys broadcast into the trainer's same-named attributes."""
    from confluid import flow, load
    from confluid.fluid import Fluid

    loaded = load(_FLAT_YAML)
    t = loaded["trainer"]
    t = flow(t) if isinstance(t, Fluid) else t

    assert t.model == "MODEL_SENTINEL"
    assert isinstance(t.train_set, Flux)  # train_set is typed Dataset[Any] — a real Dataset, not a sentinel
    assert t.val_set == "VAL_SENTINEL"
    # max_epochs broadcasts into the stored attribute...
    assert t.max_epochs == 7
    # ...batch_size/num_workers into the deferred loader stub...
    assert t.train_loader.kwargs["batch_size"] == 64
    assert t.train_loader.kwargs["num_workers"] == 4
    assert t.train_loader.kwargs["persistent_workers"] is True
    # ...and max_epochs/logger into the base's deferred L.Trainer stub at construction.
    assert t.lightning.kwargs["max_epochs"] == 7
    assert isinstance(t.lightning.kwargs["logger"], list) and len(t.lightning.kwargs["logger"]) == 1


def test_logger_broadcasts_and_flows_into_lightning() -> None:
    """`logger: [...]` broadcasts into the base's deferred L.Trainer stub; flowing it
    materializes the `!class:` logger Fluids into live instances."""
    from confluid import flow, load
    from confluid.fluid import Fluid
    from pytorch_lightning.loggers import TensorBoardLogger

    loaded = load(_FLAT_YAML)
    t = loaded["trainer"]
    t = flow(t) if isinstance(t, Fluid) else t

    # Staged on the deferred L.Trainer stub as a list with one logger Fluid...
    staged = t.lightning.kwargs["logger"]
    assert isinstance(staged, list) and len(staged) == 1
    # ...flowing the lightning stub materializes it into a live TensorBoardLogger.
    lt = flow(t.lightning)
    assert any(isinstance(lg, TensorBoardLogger) for lg in lt.loggers)
    assert all(not isinstance(lg, Fluid) for lg in lt.loggers)


# ---------------------------------------------------------------------------
# !lazy: model -> num_classes injected from the dataset at run time.
# ---------------------------------------------------------------------------


def test_lazy_model_gets_num_classes_injected_at_run(tmp_path: object) -> None:
    from confluid import Class as ConfluidClass
    from confluid import LazyClass
    from marainer.torch import collate_fn_with_metadata
    from torch.utils.data import DataLoader

    # Model declared !lazy: WITHOUT num_classes -> derived from the dataset (4).
    trainer = LightningClassifier(
        model=LazyClass(_TinyClassifier),  # no num_classes; default 3 would be wrong
        train_set=_DimsDataset(16, num_classes=4),
    )
    trainer.train_loader = ConfluidClass(DataLoader, batch_size=8, collate_fn=collate_fn_with_metadata)
    trainer.lightning = _cpu_lightning(tmp_path)
    trainer.run()
    # The lazy model was built with the derived num_classes=4 (fc out_features).
    assert isinstance(trainer.model, _TinyClassifier)
    assert trainer.model.fc.out_features == 4


# ---------------------------------------------------------------------------
# num_classes resolution: a concrete model's head is authoritative
# ---------------------------------------------------------------------------


class _NCClassifier(nn.Module):
    """Concrete classifier exposing ``num_classes`` (like TimmModel)."""

    def __init__(self, num_classes: int = 5, input_size: int = 4) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)  # type: ignore[no-any-return]


def test_concrete_model_num_classes_is_authoritative() -> None:
    # Model head sized to 5; the dataset only has 3 classes. The concrete model wins —
    # walking the data would mis-size the metrics against the head.
    trainer = LightningClassifier(
        model=_NCClassifier(num_classes=5),
        train_set=_make_samples(16, num_classes=3),
        balance_classes=False,
    )
    trainer.initialize()
    assert trainer.num_classes == 5
    assert isinstance(trainer.val_metrics, MetricCollection)
    sized = [m for m in trainer.val_metrics.values() if hasattr(m, "num_classes")]
    assert sized and all(m.num_classes == 5 for m in sized)


def test_concrete_model_num_classes_overrides_mismatched_explicit() -> None:
    # An explicit trainer num_classes that disagrees with the concrete model is overridden
    # by the model (and a warning is logged).
    trainer = LightningClassifier(
        model=_NCClassifier(num_classes=5),
        train_set=_make_samples(16, num_classes=3),
        num_classes=9,
        balance_classes=False,
    )
    trainer.initialize()
    assert trainer.num_classes == 5


# ---------------------------------------------------------------------------
# String targets: fit a LabelMap, encode the data, expose the map
# ---------------------------------------------------------------------------


def _string_samples(names: list[str], n: int = 18, input_size: int = 4) -> Flux:
    # A Flux (torch Dataset) of string-target samples — the trainer's train_set is Dataset[Any].
    return Flux(
        source=[Sample(input=torch.randn(input_size), target=names[i % len(names)], metadata={}) for i in range(n)]
    )


def test_string_targets_fit_encode_and_expose_map() -> None:
    from sampleflux import LabelMap

    names = ["dog", "cat", "bird"]  # sklearn sorts -> bird=0, cat=1, dog=2
    trainer = LightningClassifier(
        model=_TinyClassifier(num_classes=3),  # no num_classes attr -> nc from the fitted map
        train_set=_string_samples(names),
        balance_classes=False,
    )
    trainer.initialize()
    # A LabelMap was fitted and exposed.
    assert isinstance(trainer.label_map, LabelMap)
    assert trainer.label_map.label_names == ["bird", "cat", "dog"]
    assert trainer.num_classes == 3
    # The train set is now a Flux whose targets are encoded class-id ints, and it exposes
    # label_names so marainer persists class_names.json.
    from typing import Sized

    train_set = cast(Flux, trainer.train_set)
    assert train_set.label_names == ["bird", "cat", "dog"]  # type: ignore[attr-defined]
    encoded = {train_set[i].target for i in range(len(cast(Sized, train_set)))}
    assert encoded == {0, 1, 2}
    assert all(isinstance(t, int) for t in encoded)


def test_explicit_label_map_drives_encoding_and_num_classes() -> None:
    from sampleflux import LabelMap

    lm = LabelMap(mapping={"cat": 0, "dog": 1, "bird": 2, "fish": 3})  # 4 classes, pinned
    trainer = LightningClassifier(
        model=_TinyClassifier(num_classes=4),
        train_set=_string_samples(["cat", "dog", "bird"]),  # only 3 appear in train
        label_map=lm,
        balance_classes=False,
    )
    trainer.initialize()
    # num_classes comes from the pinned map (4), not the 3 labels that appear in the data.
    assert trainer.num_classes == 4
    assert trainer.label_map is lm
    train_set = cast(Flux, trainer.train_set)
    sample0 = train_set[0]
    assert sample0.target == lm.mapping[_string_samples(["cat", "dog", "bird"])[0].target]


def test_multilabel_string_targets_raise() -> None:
    samples = [Sample(input=torch.randn(4), target=["a", "b"], metadata={}) for _ in range(8)]
    trainer = LightningClassifier(model=_TinyClassifier(num_classes=2), train_set=Flux(source=samples))
    with pytest.raises(NotImplementedError):
        trainer.initialize()


# ---------------------------------------------------------------------------
# Balanced loss: inverse-frequency class weights
# ---------------------------------------------------------------------------


def test_balance_classes_injects_class_weights() -> None:
    # Imbalanced: class 0 appears 10x, classes 1/2 appear 3x each.
    samples = (
        [Sample(input=torch.randn(4), target=0, metadata={}) for _ in range(10)]
        + [Sample(input=torch.randn(4), target=1, metadata={}) for _ in range(3)]
        + [Sample(input=torch.randn(4), target=2, metadata={}) for _ in range(3)]
    )
    trainer = LightningClassifier(
        model=_TinyClassifier(num_classes=3),
        train_set=Flux(source=samples),
        num_classes=3,
        balance_classes=True,
    )
    trainer.initialize()
    assert trainer.class_weights is not None
    assert tuple(trainer.class_weights.shape) == (3,)
    # Rarer classes get the larger weight.
    assert trainer.class_weights[1] > trainer.class_weights[0]
    assert trainer.class_weights[2] > trainer.class_weights[0]
    # The weight is wired into the loss.
    weight = cast(torch.Tensor, trainer.loss_fn.weight)
    assert weight is not None
    assert torch.allclose(weight, trainer.class_weights)


def test_balance_classes_false_leaves_loss_unweighted() -> None:
    trainer = LightningClassifier(
        model=_TinyClassifier(num_classes=3),
        train_set=_make_samples(16, num_classes=3),
        num_classes=3,
        balance_classes=False,
    )
    trainer.initialize()
    assert trainer.class_weights is None
    assert trainer.loss_fn.weight is None


def test_balance_skipped_when_loss_not_weight_capable() -> None:
    # nn.MSELoss has no `weight` constructor arg → not balance-capable: the loss is left untouched
    # and class_weights stays None (a warning is logged via loggair).
    trainer = LightningClassifier(
        model=_TinyClassifier(num_classes=3),
        train_set=_make_samples(16, num_classes=3),
        num_classes=3,
        balance_classes=True,
    )
    trainer.loss_fn = nn.MSELoss()
    trainer.initialize()
    assert trainer.class_weights is None
    assert isinstance(trainer.loss_fn, nn.MSELoss)


def test_loss_accepts_weight_checks_constructor() -> None:
    # The confluid-aware balance-capability check, on both live and deferred (lazy/Fluid) losses.
    from confluid import LazyClass

    from sonair.classifier import _loss_accepts_weight

    assert _loss_accepts_weight(nn.CrossEntropyLoss()) is True  # live, weight in ctor
    assert _loss_accepts_weight(nn.MSELoss()) is False  # live, no weight arg
    assert _loss_accepts_weight(LazyClass(nn.CrossEntropyLoss)) is True  # deferred, resolved via target
    assert _loss_accepts_weight(LazyClass(nn.MSELoss)) is False  # deferred, no weight arg


def test_metrics_accepts_ready_made_collection() -> None:
    # When `metrics` is already a MetricCollection, initialize() clones it per split (no re-wrap).
    from torchmetrics.classification import MulticlassAccuracy

    coll = MetricCollection({"top1": MulticlassAccuracy(num_classes=3, top_k=1)})
    trainer = LightningClassifier(
        model=_TinyClassifier(num_classes=3),
        train_set=_make_samples(16, num_classes=3),
        num_classes=3,
        balance_classes=False,
        metrics=coll,
    )
    trainer.initialize()
    assert isinstance(trainer.train_metrics, MetricCollection)
    assert all(str(k).startswith("train/") for k in trainer.train_metrics.keys())
    assert all(str(k).startswith("val/") for k in trainer.val_metrics.keys())
    assert all(str(k).startswith("test/") for k in trainer.test_metrics.keys())


# ---------------------------------------------------------------------------
# I/O contract: trained_model @output + Mandatory model/train_set inputs
# ---------------------------------------------------------------------------
def test_declares_trained_model_output() -> None:
    from confluid import output_specs

    specs = {s["name"]: s for s in output_specs(LightningClassifier)}
    assert "trained_model" in specs
    assert specs["trained_model"]["type"] == "Module"
    # The marker rides on the getter so FluxStudio / form-spec can read it.
    fget = LightningClassifier.__dict__["trained_model"].fget
    assert getattr(fget, "__confluid_output__", False) is True


def test_inherits_computed_metrics_and_label_names_outputs() -> None:
    """The base marainer outputs surface for the classification trainer via output_specs' MRO walk."""
    from confluid import output_specs

    names = {s["name"] for s in output_specs(LightningClassifier)}
    assert {"trained_model", "computed_metrics", "label_names"} <= names


def test_trained_model_is_not_a_config_field() -> None:
    from confluid import to_pydantic

    assert "trained_model" not in to_pydantic(LightningClassifier).model_fields


def test_trained_model_raises_before_run() -> None:
    # A lazy (!lazy:) model stays a deferred Fluid until run() flows it — the output guards this.
    trainer = LightningClassifier(
        model=LazyClass(_TinyClassifier, num_classes=3),
        train_set=_make_samples(8, num_classes=3),
    )
    with pytest.raises(RuntimeError, match="only available after run"):
        _ = trainer.trained_model


def test_trained_model_returns_live_model_after_initialize() -> None:
    model = _TinyClassifier(num_classes=3)
    trainer = LightningClassifier(
        model=model,
        train_set=_make_samples(16, num_classes=3),
        balance_classes=False,
    )
    trainer.initialize()  # flows the model; a concrete model passes through
    assert isinstance(trainer.trained_model, nn.Module)
    assert trainer.trained_model is trainer.model


def test_model_is_mandatory_input() -> None:
    # Only `model` is mandatory now — train_set/test_set are optional (predict/test need only a model
    # + test_set), so a single class serves fit/evaluate/test/predict.
    from confluid import input_specs, mandatory_param_names

    assert mandatory_param_names(LightningClassifier) == {"model"}
    required = {s["name"] for s in input_specs(LightningClassifier) if s["required"]}
    assert required == {"model"}
