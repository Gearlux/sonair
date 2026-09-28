"""Tests for LightningClassifier."""

from typing import Any, List, cast, get_args

import numpy as np
import pytest
import torch
import torch.nn as nn
from confluid import PartialClass
from confluid.fluid import Fluid
from recordstream import Image, Label, Record, Stream, inverse_frequency_weights
from recordstream.runnable import runnable_entrypoints
from torchmetrics import MetricCollection

from sonair.lightning_classifier import ClassifierTask, LightningClassifier


def _stub(slot: Any) -> Fluid:
    """The deferred Fluid behind a ``Partial[T]`` slot — asserts it is STILL deferred, and narrows.

    ``Partial[T]`` is ``Union[T, Fluid]``, so reading ``.kwargs`` needs the Fluid arm. Going through
    this helper means every such read also asserts the slot has not been eagerly built, which is
    the property these tests actually care about.
    """
    assert isinstance(slot, Fluid), f"expected a deferred stub, got {type(slot).__name__}"
    return slot


def _ts(inp: Any, tgt: Any, **meta: Any) -> Record:
    """Build a record dict: raw tensor under "image", a Label target under "class", plain meta keys."""
    record: Record = {"image": inp, "class": Label(tgt)}
    record.update(meta)
    return record


def _stack_collate(batch: list) -> Record:
    """A test-only collate that stacks raw feature/target tensors.

    These synthetic trainer tests use a tiny ``nn.Linear`` model over 1-D feature vectors carried as
    BARE-tensor ``"image"`` values, which ``recordstream.collate_records`` batches as a per-record LIST
    (a plain value); this shim pre-stacks them so the steps receive one ``[N, F]`` tensor. Real image
    records carry item-typed values and use ``collate_records`` (the default loader collate + the
    ``test_classification`` / smoke tests); this only bridges the non-image synthetic fixtures."""
    xs = torch.stack([torch.as_tensor(r["image"]) for r in batch])
    ys = torch.stack(
        [torch.as_tensor(r["class"].value if isinstance(r["class"], Label) else r["class"]) for r in batch]
    )
    metas: list = [
        {k: (v.value if isinstance(v, Label) else v) for k, v in r.items() if k not in ("image", "class")}
        for r in batch
    ]
    return {"image": xs, "class": Label(value=ys), "metadata": metas}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _TinyClassifier(nn.Module):
    def __init__(self, num_classes: int = 3, input_size: int = 4) -> None:
        super().__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)  # type: ignore[no-any-return]


def _make_batch(n: int = 8, num_classes: int = 3, input_size: int = 4) -> Record:
    # A batched record shaped like collate_records output: the input Image holds the
    # (pre-batched) feature tensor as an ndarray (an Image can't hold a live tensor), the target
    # Label holds the [N] int64 tensor, and per-record metadata rides a "metadata" key.
    x = torch.randn(n, input_size)
    y = torch.randint(0, num_classes, (n,))
    return {
        "image": Image(x.numpy(), layout="CHW"),
        "class": Label(value=y),
        "metadata": [{"idx": i} for i in range(n)],
    }


def _make_records(n: int = 16, num_classes: int = 3, input_size: int = 4) -> Stream:
    # A Stream (torch Dataset) — the trainer's train_set is typed Dataset[Any].
    return Stream(source=[_ts(torch.randn(input_size), i % num_classes) for i in range(n)])


# ---------------------------------------------------------------------------
# LightningClassifier defaults
# ---------------------------------------------------------------------------


def test_loss_fn_defaults_to_cross_entropy() -> None:
    """The default is a SENTINEL until `initialize()` knows the target shape.

    A live `CrossEntropyLoss` here could not be told apart from a user's explicit one, so a
    multi-label config would silently train with the wrong loss.
    """
    trainer = LightningClassifier(
        model=_TinyClassifier(num_classes=3),
        train_set=_make_records(16),
        num_classes=3,
    )
    assert trainer.loss_fn is None
    trainer.initialize()
    assert isinstance(trainer.loss_fn, nn.CrossEntropyLoss)


def test_an_explicitly_wired_loss_is_never_replaced() -> None:
    """The sentinel distinguishes "not chosen" from "chosen" — a wired loss survives."""
    chosen = nn.MultiLabelSoftMarginLoss()
    trainer = LightningClassifier(
        model=_TinyClassifier(num_classes=3), train_set=_make_records(16), num_classes=3, balance_classes=False
    )
    trainer.loss_fn = chosen
    trainer.initialize()
    assert trainer.loss_fn is chosen


def test_initialize_wires_metrics_with_correct_prefixes() -> None:
    trainer = LightningClassifier(
        model=_TinyClassifier(num_classes=3),
        train_set=_make_records(16),
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
        train_set=_make_records(16, num_classes=10),
        num_classes=10,
    )
    trainer.initialize()
    assert trainer.train_metrics is not None
    keys = [str(k) for k in trainer.train_metrics.keys()]
    assert not any("top5" in k for k in keys), f"top5 should be opt-in, got: {keys}"


def test_top5_opt_in_via_metrics_dict() -> None:
    # top-5 is opt-in: add it to the `metrics` dict; initialize() flows it with num_classes.
    from confluid import PartialClass
    from torchmetrics.classification import MulticlassAccuracy

    trainer = LightningClassifier(
        model=_TinyClassifier(num_classes=10),
        train_set=_make_records(16, num_classes=10),
        num_classes=10,
        metrics={
            "accuracy_top1": PartialClass(MulticlassAccuracy, top_k=1),
            "accuracy_top5": PartialClass(MulticlassAccuracy, top_k=5),
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
        train_set=_make_records(16),
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
        train_set=_make_records(16),
        num_classes=3,
    )
    batch = _make_batch(n=8, num_classes=3)
    loss = trainer.validation_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0


def test_metrics_update_on_training_step() -> None:
    trainer = LightningClassifier(
        model=_TinyClassifier(num_classes=3),
        train_set=_make_records(16),
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
    """Map-style torch Dataset of integer-target records."""

    def __init__(self, n: int, num_classes: int, input_size: int = 4) -> None:
        self._s = [_ts(torch.randn(input_size), torch.tensor(i % num_classes, dtype=torch.int64)) for i in range(n)]
        self._nc = num_classes

    def __len__(self) -> int:
        return len(self._s)

    def __getitem__(self, i: int) -> Record:
        return self._s[i]

    def derived_dimensions(self) -> dict:
        return {"num_classes": self._nc}


def test_omitted_num_classes_leaves_metrics_none_until_run() -> None:
    """With num_classes omitted, metrics stay None post-construction — sonair builds them
    in run() once it derives the count (see test_omitted_num_classes_derived_and_built_at_run),
    rather than carrying a deferred Partial fluid."""
    trainer = LightningClassifier(
        model=_TinyClassifier(num_classes=4),
        train_set=_make_records(16, num_classes=4),
    )
    assert trainer.num_classes is None
    assert trainer.train_metrics is None
    assert trainer.val_metrics is None
    assert trainer.test_metrics is None


def _cpu_lightning(tmp_path: object) -> Any:
    """A test-friendly Lightning Trainer Target stub (cpu, 1 batch, no logger)."""
    import pytorch_lightning as L
    from confluid import Target as ConfluidClass

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
    from confluid import Target as ConfluidClass
    from torch.utils.data import DataLoader

    # Minimal constructor; everything else configured post-construction (the paradigm).
    trainer = LightningClassifier(
        model=_TinyClassifier(num_classes=4),
        train_set=_DimsDataset(16, num_classes=4),
        # num_classes omitted -> derived from the dataset
    )
    trainer.train_loader = ConfluidClass(DataLoader, batch_size=8, collate_fn=_stack_collate)
    trainer.lightning = _cpu_lightning(tmp_path)
    trainer.run()
    assert isinstance(trainer.val_metrics, MetricCollection)
    sized = [m for m in trainer.val_metrics.values() if hasattr(m, "num_classes")]
    assert sized and all(m.num_classes == 4 for m in sized)


def test_default_loader_uses_collate_records() -> None:
    from confluid.fluid import Target as ConfluidClass
    from recordstream import collate_records

    trainer = LightningClassifier(
        model=_TinyClassifier(num_classes=3),
        train_set=_make_records(16),
        num_classes=3,
    )
    # The train_loader is a deferred Target stub; check its kwargs contain the collate_fn.
    assert isinstance(trainer.train_loader, ConfluidClass)
    assert _stub(trainer.train_loader).kwargs.get("collate_fn") is collate_records


# ---------------------------------------------------------------------------
# Ergonomic knobs (batch_size / num_workers / persist_workers / max_epochs) are
# defaulted attributes folded into the deferred loader + Lightning stubs at run().
# ---------------------------------------------------------------------------


def test_ergonomic_knobs_reach_loader_and_lightning_stubs() -> None:
    from confluid import Target as ConfluidClass

    # batch_size / num_workers are ctor knobs baked into the deferred loader stub in
    # __init__ (the constructor stays side-effect-free — PartialClass defers construction).
    trainer = LightningClassifier(
        model=_TinyClassifier(num_classes=3),
        train_set=_make_records(16),
        num_classes=3,
        batch_size=64,
        num_workers=4,
        max_epochs=7,
    )
    tl = _stub(trainer.train_loader).kwargs
    assert tl["batch_size"] == 64 and tl["num_workers"] == 4
    assert tl["shuffle"] is True and tl["persistent_workers"] is True
    assert _stub(trainer.val_loader).kwargs["shuffle"] is False
    # max_epochs is baked into the base's deferred L.Trainer stub at construction.
    assert isinstance(trainer.lightning, ConfluidClass)
    assert _stub(trainer.lightning).kwargs["max_epochs"] == 7


def test_persistent_workers_guarded_when_num_workers_zero() -> None:
    # persistent_workers=True is meaningless (and raises in torch) when num_workers=0,
    # so __init__ derives it as `num_workers != 0`.
    trainer = LightningClassifier(
        model=_TinyClassifier(num_classes=3),
        train_set=_make_records(16),
        num_classes=3,
        num_workers=0,
    )
    assert _stub(trainer.train_loader).kwargs["persistent_workers"] is False


def test_persistent_workers_can_be_named_instead_of_derived() -> None:
    """The HOST-scoped knob (`!scope:os=darwin` -> `persistent_workers: false`): the workers
    stay, only their persistence goes. macOS terminates them slowly enough that a short run
    spends longer stopping than training, which no derivation from `num_workers` can see."""
    trainer = LightningClassifier(
        model=_TinyClassifier(num_classes=3),
        train_set=_make_records(16),
        num_classes=3,
        num_workers=2,
        persistent_workers=False,
    )
    for split in ("train_loader", "val_loader", "test_loader"):
        kwargs = _stub(getattr(trainer, split)).kwargs
        assert kwargs["persistent_workers"] is False, split
        assert kwargs["num_workers"] == 2, split


def test_explicit_loader_override_is_respected() -> None:
    from confluid import Target as ConfluidClass
    from torch.utils.data import DataLoader

    # __init__ bakes the ctor batch_size into the default loader stub...
    trainer = LightningClassifier(
        model=_TinyClassifier(num_classes=3),
        train_set=_make_records(16),
        num_classes=3,
        batch_size=8,
    )
    assert _stub(trainer.train_loader).kwargs["batch_size"] == 8
    # ...but a post-construction explicit loader replaces it wholesale (nothing folds
    # over it at run() — the base run() flows whatever loader is wired).
    trainer.train_loader = ConfluidClass(DataLoader, batch_size=999)
    assert _stub(trainer.train_loader).kwargs["batch_size"] == 999


# ---------------------------------------------------------------------------
# Flat-config broadcasting + logger deferred-construction (the user's ask)
# ---------------------------------------------------------------------------


_FLAT_YAML = """
trainer: !class:sonair.LightningClassifier
  run_name: x
model: "MODEL_SENTINEL"
train_set: !class:recordstream.Stream()
val_set: "VAL_SENTINEL"
batch_size: 64
num_workers: 4
max_epochs: 7
trackers:
  - !class:matrainer.CsvTracker
    experiment_name: bcast
    save_dir: ./_sonair_test_logs
"""


def test_flat_broadcasting_wires_trainer_attributes_from_top_level_keys() -> None:
    """Top-level sibling keys broadcast into the trainer's same-named attributes."""
    from confluid import flow, load
    from confluid.fluid import Fluid

    loaded = load(_FLAT_YAML)
    t = loaded["trainer"]
    t = flow(t) if isinstance(t, Fluid) else t

    assert t.model == "MODEL_SENTINEL"
    # `train_set` is declared `Partial[RecordSource]`, so the slot keeps its marker until flowed —
    # what this asserts is that the real dataset reached it, not a "VAL_SENTINEL"-style string.
    assert isinstance(flow(_stub(t.train_set)), Stream)
    assert t.val_set == "VAL_SENTINEL"
    # max_epochs broadcasts into the stored attribute...
    assert t.max_epochs == 7
    # ...batch_size/num_workers into the deferred loader stub...
    assert _stub(t.train_loader).kwargs["batch_size"] == 64
    assert _stub(t.train_loader).kwargs["num_workers"] == 4
    assert _stub(t.train_loader).kwargs["persistent_workers"] is True
    # ...and max_epochs into the deferred L.Trainer stub at construction. `logger` is NOT
    # staged there any more: the trackers become the Trainer's loggers in `_build_trainer`,
    # because a tracker may still be a deferred Fluid while this stub is being built.
    assert _stub(t.lightning).kwargs["max_epochs"] == 7
    assert "logger" not in _stub(t.lightning).kwargs
    assert t.trackers is not None and len(list(t.trackers)) == 1


def test_a_list_valued_trackers_key_reaches_the_slot() -> None:
    """The property the whole `trackers` slot shape exists for.

    Confluid gates LIST broadcast on the receiving slot's ANNOTATION, so a `trackers: [...]`
    key only lands because the constructor param is typed `Union[Tracker, Iterable[Tracker]]`.
    Typed `Optional[List[Tracker]]` and assigned after construction — its previous shape — the
    same key silently did nothing, and every wired tracker was ignored by a run that otherwise
    looked fine.
    """
    from confluid import flow, load
    from confluid.fluid import Fluid

    t = load(_FLAT_YAML)["trainer"]
    t = flow(t) if isinstance(t, Fluid) else t

    assert t.trackers is not None, "a list-valued `trackers:` key did not reach the slot"
    assert len(list(t.trackers)) == 1


def test_the_constructor_does_not_touch_the_trackers_it_was_given() -> None:
    """Partial-init: storing VERBATIM is what keeps a deferred marker deferred.

    Normalizing in `__init__` would iterate the value, and a `Fluid` is not iterable —
    `list(Target(...))` raises `'Target' object is not iterable`.
    """
    from confluid.fluid import Target

    marker = Target("matrainer.CsvTracker")
    trainer = LightningClassifier(model=_TinyClassifier(), trackers=marker)
    assert trainer.trackers is marker  # untouched, not wrapped, not built


def test_the_trackers_become_the_trainers_loggers() -> None:
    """There is no `logger:` slot any more — the trackers ARE the metric sinks."""
    from matrainer import CsvTracker
    from matrainer.lightning import TrackerLogger

    tracker = CsvTracker(experiment_name="e", run_name="r", save_dir="./_sonair_test_logs")
    trainer = LightningClassifier(model=_TinyClassifier(), trackers=[tracker])
    built = trainer._build_trainer()

    adapters = [lg for lg in built.loggers if isinstance(lg, TrackerLogger)]
    assert len(adapters) == 1
    assert adapters[0].tracker is tracker


def test_a_metric_names_key_broadcasts_from_a_flat_config() -> None:
    """The knob has to be reachable the way every other one is — a top-level sibling key.

    Wired in a CONSTRUCTOR slot it arrives as a deferred marker (only `!class:X()` builds
    eagerly), which is why the slot stores it verbatim and `as_metric_names` flows it at use.
    """
    from confluid import flow, load
    from confluid.fluid import Fluid
    from matrainer.metric_names import as_metric_names

    yaml = """
trainer: !class:sonair.LightningClassifier
  run_name: x
model: "MODEL_SENTINEL"
metric_names: !class:matrainer.MetricNames
  target: keras
"""
    trainer = load(yaml)["trainer"]
    trainer = flow(trainer) if isinstance(trainer, Fluid) else trainer

    assert trainer.metric_names is not None, "a `metric_names:` key did not reach the slot"
    assert as_metric_names(trainer.metric_names).target == "keras"


def test_the_metric_names_knob_reaches_the_adapters() -> None:
    """One `metric_names:` key, and every tracker this run writes to records that vocabulary."""
    from matrainer import CsvTracker, MetricNames
    from matrainer.lightning import TrackerLogger

    names = MetricNames(target="keras")
    trainer = LightningClassifier(
        model=_TinyClassifier(),
        trackers=[CsvTracker(experiment_name="e", run_name="r", save_dir="./_sonair_test_logs")],
        metric_names=names,
    )

    adapters = [lg for lg in trainer._build_trainer().loggers if isinstance(lg, TrackerLogger)]
    assert adapters and adapters[0].metric_names is names


def test_the_training_loss_is_recorded_under_the_canonical_name() -> None:
    """`train/loss`, never Lightning's `train/loss_epoch`.

    Lightning appends `_step` / `_epoch` when ONE name carries both, so the single
    `on_step=True, on_epoch=True` call this used to make recorded `train/loss_epoch` — a name
    no other backend writes, in the column a cross-engine sweep compares. The per-step series
    is still logged, under a name of its own.
    """
    logged: List[str] = []
    trainer = LightningClassifier(model=_TinyClassifier(num_classes=3), num_classes=3)
    trainer.log = lambda name, *_args, **_kw: logged.append(name)  # type: ignore[method-assign]
    trainer._resolve_task_defaults()

    trainer.training_step(_make_batch(), 0)

    assert "train/loss" in logged
    assert "train/loss_step" in logged
    assert not any(name.endswith("_epoch") for name in logged)


def test_the_default_metric_set_names_accuracy_what_every_backend_can_name_it() -> None:
    """`accuracy`, not `accuracy_top1`.

    The other seven backends get their accuracy metric's name from their own engine — fastai
    from `fn.__name__`, Keras from a metric's `name=` — and none of them can produce `_top1`.
    A different word here would put one quantity in two columns of the same sweep report.
    `top_k=1` lives in the template, where it is checkable.
    """
    trainer = LightningClassifier(model=_TinyClassifier(num_classes=3), num_classes=3)
    trainer._resolve_task_defaults()

    assert set(trainer.metrics) == {"accuracy", "f1_macro"}
    assert trainer.metrics["accuracy"].kwargs["top_k"] == 1


def test_flowing_twice_does_not_rebuild_a_deferred_tracker() -> None:
    """A re-flowed `Fluid` would be a SECOND tracker — two MLflow runs for one training run."""
    from confluid.fluid import Target

    trainer = LightningClassifier(
        model=_TinyClassifier(), trackers=[Target("matrainer.CsvTracker", save_dir="./_sonair_test_logs")]
    )
    first = trainer._live_trackers()
    second = trainer._live_trackers()
    assert first[0] is second[0]


# ---------------------------------------------------------------------------
# !lazy: model -> num_classes injected from the dataset at run time.
# ---------------------------------------------------------------------------


def test_lazy_model_gets_num_classes_injected_at_run(tmp_path: object) -> None:
    from confluid import PartialClass
    from confluid import Target as ConfluidClass
    from torch.utils.data import DataLoader

    # Model declared !lazy: WITHOUT num_classes -> derived from the dataset (4).
    trainer = LightningClassifier(
        model=PartialClass(_TinyClassifier),  # no num_classes; default 3 would be wrong
        train_set=_DimsDataset(16, num_classes=4),
    )
    trainer.train_loader = ConfluidClass(DataLoader, batch_size=8, collate_fn=_stack_collate)
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
        train_set=_make_records(16, num_classes=3),
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
        train_set=_make_records(16, num_classes=3),
        num_classes=9,
        balance_classes=False,
    )
    trainer.initialize()
    assert trainer.num_classes == 5


# ---------------------------------------------------------------------------
# String targets: fit a LabelMap, encode the data, expose the map
# ---------------------------------------------------------------------------


def _string_records(names: list[str], n: int = 18, input_size: int = 4) -> Stream:
    # A Stream (torch Dataset) of string-target records — the trainer's train_set is Dataset[Any].
    return Stream(source=[_ts(torch.randn(input_size), names[i % len(names)]) for i in range(n)])


def test_string_targets_fit_encode_and_expose_map() -> None:
    from recordstream import LabelMap

    names = ["dog", "cat", "bird"]  # sklearn sorts -> bird=0, cat=1, dog=2
    trainer = LightningClassifier(
        model=_TinyClassifier(num_classes=3),  # no num_classes attr -> nc from the fitted map
        train_set=_string_records(names),
        balance_classes=False,
    )
    trainer.initialize()
    # A LabelMap was fitted and exposed.
    assert isinstance(trainer.label_map, LabelMap)
    assert trainer.label_map.class_names == ["bird", "cat", "dog"]
    assert trainer.num_classes == 3
    # The train set is now a Stream whose targets are encoded class-id ints, and it exposes
    # class_names so matrainer persists class_names.json.
    from typing import Sized

    from recordstream import iter_key

    train_set = cast(Stream, trainer.train_set)
    assert train_set.class_names == ["bird", "cat", "dog"]
    assert len(cast(Sized, train_set))  # touch len for parity with the old index-walk
    encoded = set(iter_key(train_set, "class"))
    assert encoded == {0, 1, 2}
    assert all(isinstance(t, int) for t in encoded)


def test_explicit_label_map_drives_encoding_and_num_classes() -> None:
    from recordstream import LabelMap

    lm = LabelMap(mapping={"cat": 0, "dog": 1, "bird": 2, "fish": 3})  # 4 classes, pinned
    trainer = LightningClassifier(
        model=_TinyClassifier(num_classes=4),
        train_set=_string_records(["cat", "dog", "bird"]),  # only 3 appear in train
        label_map=lm,
        balance_classes=False,
    )
    trainer.initialize()
    # num_classes comes from the pinned map (4), not the 3 labels that appear in the data.
    assert trainer.num_classes == 4
    assert trainer.label_map is lm
    from recordstream import iter_key

    train_set = cast(Stream, trainer.train_set)
    raw_name = next(iter(iter_key(_string_records(["cat", "dog", "bird"]), "class")))
    assert next(iter(iter_key(train_set, "class"))) == lm.mapping[raw_name]


def test_multilabel_string_targets_fit_encode_and_train() -> None:
    """Multi-label targets: names fitted to ids, BCE + Multilabel metrics chosen automatically."""
    from recordstream import MultiLabel

    records = [{"image": torch.randn(4), "class": MultiLabel(["a", "b"])} for _ in range(8)]
    trainer = LightningClassifier(
        model=_TinyClassifier(num_classes=2), train_set=Stream(source=records), balance_classes=False
    )
    trainer.initialize()

    assert trainer.multilabel is True
    assert isinstance(trainer.loss_fn, nn.BCEWithLogitsLoss)  # not CrossEntropy
    assert trainer.num_classes == 2
    assert trainer.label_map is not None and trainer.label_map.mapping == {"a": 0, "b": 1}


def test_the_classifier_picks_the_multi_hot_target_when_multilabel() -> None:
    """The task-shaped choice `_targets` makes: [N, C] float multi-hot, not [N] int64 ids.

    The multi-hot ITSELF is `recordstream.multi_hot` (pinned there, and numpy so a non-torch
    backend reuses it) — what this pins is that the classifier selects it.
    """
    from recordstream import MultiLabel, collate_records

    trainer = LightningClassifier(model=_TinyClassifier(num_classes=3), num_classes=3)
    trainer.multilabel = True
    batch = collate_records([{"class": MultiLabel([0, 2])}, {"class": MultiLabel([1])}])

    y = trainer._targets(batch)

    assert y.shape == (2, 3)
    assert y.dtype is torch.float32
    assert y.tolist() == [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]


def test_the_classifier_names_int64_for_single_label_targets() -> None:
    """int32 label tensors are legal; CrossEntropyLoss rejects them, so the dtype is named."""
    from recordstream import Label, collate_records

    trainer = LightningClassifier(model=_TinyClassifier(num_classes=3), num_classes=3)
    batch = collate_records([{"class": Label(torch.tensor(i, dtype=torch.int32))} for i in range(3)])

    y = trainer._targets(batch)

    assert y.dtype is torch.int64 and y.tolist() == [0, 1, 2]
    nn.CrossEntropyLoss()(torch.randn(3, 3), y)  # would raise on int32


def test_multilabel_fit_runs_end_to_end(tmp_path: object) -> None:
    """The whole multi-label path: MultiLabel targets -> multi-hot -> BCE -> Multilabel metrics.

    The unit tests above pin each seam; this proves they compose — a `[N, C]` float target
    against `[N, C]` logits is exactly what a shape mismatch would blow up on.
    """
    from confluid import Target as ConfluidClass
    from recordstream import MultiLabel, collate_records
    from torch.utils.data import DataLoader

    records = [{"image": torch.randn(4), "class": MultiLabel([i % 3, (i + 1) % 3])} for i in range(16)]
    trainer = LightningClassifier(
        model=_TinyClassifier(num_classes=3), train_set=Stream(source=records), balance_classes=False
    )
    trainer.train_loader = ConfluidClass(DataLoader, batch_size=8, collate_fn=collate_records)
    trainer.lightning = _cpu_lightning(tmp_path)
    trainer.run()

    assert trainer.multilabel is True
    assert isinstance(trainer.loss_fn, nn.BCEWithLogitsLoss)
    sized = [m for m in trainer.val_metrics.values() if hasattr(m, "num_labels")]
    assert sized and all(m.num_labels == 3 for m in sized)


def test_multilabel_metrics_are_sized_by_num_labels() -> None:
    """torchmetrics sizes multi-label metrics by `num_labels`, multi-class by `num_classes`."""
    trainer = LightningClassifier(model=_TinyClassifier(num_classes=3), num_classes=3)
    trainer.multilabel = True
    assert trainer._metric_flow_dims() == {"num_labels": 3}
    trainer.multilabel = False
    assert trainer._metric_flow_dims() == {"num_classes": 3}


def test_multilabel_class_weights_count_every_label() -> None:
    """Balancing flattens a MultiLabel's ids — a record with two labels counts for both."""
    from recordstream import MultiLabel

    records = [{"image": torch.randn(4), "class": MultiLabel([0, 1])} for _ in range(6)]
    records += [{"image": torch.randn(4), "class": MultiLabel([2])} for _ in range(2)]
    trainer = LightningClassifier(model=_TinyClassifier(num_classes=3), train_set=Stream(source=records), num_classes=3)
    trainer.loss_fn = nn.BCEWithLogitsLoss()  # weight-capable
    trainer.initialize()

    assert trainer.class_weights is not None
    # class 2 appears twice, classes 0/1 six times each -> the rare class weighs more
    assert trainer.class_weights[2] > trainer.class_weights[0]


# ---------------------------------------------------------------------------
# Balanced loss: inverse-frequency class weights
# ---------------------------------------------------------------------------


def test_balance_classes_injects_class_weights() -> None:
    # Imbalanced: class 0 appears 10x, classes 1/2 appear 3x each.
    records = (
        [_ts(torch.randn(4), 0) for _ in range(10)]
        + [_ts(torch.randn(4), 1) for _ in range(3)]
        + [_ts(torch.randn(4), 2) for _ in range(3)]
    )
    trainer = LightningClassifier(
        model=_TinyClassifier(num_classes=3),
        train_set=Stream(source=records),
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
    weight = cast(torch.Tensor, cast(nn.Module, trainer.loss_fn).weight)
    assert weight is not None
    assert torch.allclose(weight, trainer.class_weights)


def test_balance_classes_false_leaves_loss_unweighted() -> None:
    trainer = LightningClassifier(
        model=_TinyClassifier(num_classes=3),
        train_set=_make_records(16, num_classes=3),
        num_classes=3,
        balance_classes=False,
    )
    trainer.initialize()
    assert trainer.class_weights is None
    assert cast(nn.Module, trainer.loss_fn).weight is None


def test_balance_skipped_when_loss_not_weight_capable() -> None:
    # nn.MSELoss has no `weight` constructor arg → not balance-capable: the loss is left untouched
    # and class_weights stays None (a warning is logged via loggair).
    trainer = LightningClassifier(
        model=_TinyClassifier(num_classes=3),
        train_set=_make_records(16, num_classes=3),
        num_classes=3,
        balance_classes=True,
    )
    trainer.loss_fn = nn.MSELoss()
    trainer.initialize()
    assert trainer.class_weights is None
    assert isinstance(trainer.loss_fn, nn.MSELoss)


def test_the_weight_STATISTIC_comes_from_recordstream() -> None:
    """The split: the numbers are a property of the DATA, the injection of the FRAMEWORK.

    sonair must not carry its own inverse-frequency arithmetic — that is a label statistic
    (`recordstream.inverse_frequency_weights`, pinned in recordstream's own suite). What stays
    here is the backend hook that knows what `weight=` means.
    """
    import inspect

    from recordstream import inverse_frequency_weights

    from sonair import classification_base

    assert classification_base.inverse_frequency_weights is inverse_frequency_weights
    source = inspect.getsource(classification_base)
    assert "def inverse_frequency_weights" not in source, "the arithmetic must not be re-derived here"


def test_balancing_hooks_are_overridable_per_framework() -> None:
    """`accepts_class_weights` / `apply_class_weights` are the per-backend extension point.

    A non-torch backend overrides them (Keras takes `class_weight` on `fit()`, so it never
    touches the loss). Both must therefore be public METHODS on the runnable, not module-level
    functions baked into the torch path.
    """
    calls: list = []

    class _KerasLike(LightningClassifier):
        def accepts_class_weights(self) -> bool:
            return True

        def apply_class_weights(self, weights: Any) -> None:
            calls.append(list(weights))  # a Keras backend would stash {class_id: weight} here

    records = [_ts(torch.randn(4), 0) for _ in range(6)] + [_ts(torch.randn(4), 1)]
    trainer = _KerasLike(
        model=_TinyClassifier(num_classes=2), train_set=Stream(source=records), num_classes=2, balance_classes=True
    )
    trainer.initialize()

    assert len(calls) == 1
    # The rare class weighs more — the vector came from recordstream, unchanged by the hook.
    assert calls[0][1] > calls[0][0]
    # The torch loss was NOT touched: this backend overrode the injection.
    assert getattr(trainer.loss_fn, "weight", None) is None


def test_the_torch_hook_converts_the_numpy_vector_to_a_tensor() -> None:
    """recordstream returns numpy (framework-neutral); making it a Tensor is the hook's job."""
    records = [_ts(torch.randn(4), 0) for _ in range(6)] + [_ts(torch.randn(4), 1)]
    trainer = LightningClassifier(
        model=_TinyClassifier(num_classes=2),
        train_set=Stream(source=records),
        num_classes=2,
        balance_classes=True,
    )
    trainer.initialize()

    assert isinstance(trainer.class_weights, torch.Tensor)
    assert isinstance(inverse_frequency_weights([0, 0, 1], 2), np.ndarray)


def test_metrics_accepts_ready_made_collection() -> None:
    # When `metrics` is already a MetricCollection, initialize() clones it per split (no re-wrap).
    from torchmetrics.classification import MulticlassAccuracy

    coll = MetricCollection({"top1": MulticlassAccuracy(num_classes=3, top_k=1)})
    trainer = LightningClassifier(
        model=_TinyClassifier(num_classes=3),
        train_set=_make_records(16, num_classes=3),
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
# Non-scalar metrics: a confusion matrix in the metric set
# ---------------------------------------------------------------------------
def _confusion_trainer(num_classes: int = 3) -> LightningClassifier:
    """A classifier whose metric set carries one scalar metric AND one N×N confusion matrix."""
    from torchmetrics import Metric
    from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix

    return LightningClassifier(
        model=_TinyClassifier(num_classes=num_classes),
        train_set=_make_records(16, num_classes=num_classes),
        num_classes=num_classes,
        balance_classes=False,
        metrics={
            "accuracy_top1": PartialClass[Metric](MulticlassAccuracy, top_k=1),
            "confusion": PartialClass[Metric](MulticlassConfusionMatrix),
        },
    )


def test_a_confusion_matrix_in_the_metric_set_does_not_kill_the_val_loop(tmp_path: object) -> None:
    """The counterfactual this whole path exists for — it used to raise on the first val epoch.

    `self.log_dict(<MetricCollection>)` hands Lightning the computed values, and its
    `convert_tensors_to_scalars` raises *"does not contain a single element"* on a `[C, C]`
    matrix. Measured before the fix; this asserts the run now completes AND that the matrix
    really was computed (not quietly dropped from the collection).
    """
    from confluid import Target as ConfluidClass
    from torch.utils.data import DataLoader

    trainer = _confusion_trainer()
    trainer.val_set = _make_records(8, num_classes=3)
    trainer.train_loader = ConfluidClass(DataLoader, batch_size=8, collate_fn=_stack_collate)
    trainer.val_loader = ConfluidClass(DataLoader, batch_size=8, collate_fn=_stack_collate)
    trainer.lightning = _cpu_lightning(tmp_path)
    trainer.run()

    matrix = trainer.val_metrics["val/confusion"].compute()
    assert matrix.shape == (3, 3)
    assert int(matrix.sum()) == 8  # every validation record landed in a cell


def test_log_dict_receives_the_scalars_only() -> None:
    """The split is by the COMPUTED value, so the matrix never reaches the metric logger."""
    logged: List[Any] = []
    trainer = _confusion_trainer()
    trainer.initialize()
    trainer.val_metrics.update(torch.randn(8, 3), torch.randint(0, 3, (8,)))
    trainer.log_dict = lambda payload, **_: logged.append(payload)  # type: ignore[method-assign]

    trainer._report_epoch_metrics(trainer.val_metrics)

    assert list(logged[0]) == ["val/accuracy_top1"]
    assert isinstance(logged[0]["val/accuracy_top1"], float)


def test_the_confusion_matrix_is_rendered_at_validation_epoch_end(monkeypatch: pytest.MonkeyPatch) -> None:
    """It is reported instead — as a table, through matrainer's renderer.

    Since the 2026-08-06 consolidation the reporting loop is matrainer's `report_matrices`
    (called via the task's `_report_nonscalars` hook), which reads the renderer as a module
    global of `matrainer.tracking.render` — so that is the module to patch.
    """
    from matrainer.tracking import render as mod

    rendered: List[Any] = []
    monkeypatch.setattr(mod, "render_matrix_as_table", lambda name, matrix, names: rendered.append((name, names)))

    trainer = _confusion_trainer()
    trainer.initialize()
    trainer.val_metrics.update(torch.randn(8, 3), torch.randint(0, 3, (8,)))
    trainer.log_dict = lambda payload, **_: None  # type: ignore[method-assign]
    trainer._report_epoch_metrics(trainer.val_metrics)

    assert rendered == [("val/confusion", ["0", "1", "2"])]  # integer targets -> index labels


def test_nothing_is_rendered_during_the_sanity_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lightning's sanity pass runs two batches through an untrained model — not worth a table."""
    from matrainer.tracking import render as mod

    rendered: List[Any] = []
    monkeypatch.setattr(mod, "render_matrix_as_table", lambda *a: rendered.append(a))

    trainer = _confusion_trainer()
    trainer.initialize()
    trainer.val_metrics.update(torch.randn(8, 3), torch.randint(0, 3, (8,)))
    trainer.log_dict = lambda payload, **_: None  # type: ignore[method-assign]
    trainer._trainer = cast(Any, type("_T", (), {"sanity_checking": True})())
    trainer._report_epoch_metrics(trainer.val_metrics)

    assert rendered == []


def test_eval_metrics_reset_at_epoch_START_so_the_final_state_survives_the_run() -> None:
    """Two epochs must not accumulate — and the last one must still be computable afterwards.

    Handing the collection to `log_dict` used to make Lightning reset it at epoch END, which is
    why `_write_metrics_sidecar` / `computed_metrics` (both `compute()` AFTER the run) saw an
    empty metric. Resetting at epoch start does both jobs with one call.
    """
    trainer = _confusion_trainer()
    trainer.initialize()
    trainer.log_dict = lambda payload, **_: None  # type: ignore[method-assign]

    for _ in range(2):
        trainer.on_validation_epoch_start()
        trainer.val_metrics.update(torch.randn(8, 3), torch.randint(0, 3, (8,)))
        trainer.on_validation_epoch_end()

    assert int(trainer.val_metrics["val/confusion"].compute().sum()) == 8  # one epoch, not two


def test_a_rendered_matrix_reaches_every_wired_tracker() -> None:
    """The log is not the only destination — otherwise a tracked run records the scalars only.

    `log_text` is the channel that exists for a value a `{name: float}` metric call cannot
    carry; this asserts the classifier actually uses it, and hands over the SAME table it logs.
    """

    class _Recorder:
        def __init__(self) -> None:
            self.texts: List[Any] = []

        def log_text(self, name: str, text: str, step: Any = None) -> None:
            self.texts.append((name, text, step))

    tracker = _Recorder()
    trainer = _confusion_trainer()
    trainer.trackers = cast(Any, [tracker])
    trainer.initialize()
    trainer.val_metrics.update(torch.randn(8, 3), torch.randint(0, 3, (8,)))
    trainer.log_dict = lambda payload, **_: None  # type: ignore[method-assign]

    trainer._report_epoch_metrics(trainer.val_metrics)

    assert len(tracker.texts) == 1
    name, table, step = tracker.texts[0]
    assert name == "val/confusion"
    assert table.startswith("val/confusion:") and "pred:0" in table  # the rendered table, not the tensor
    assert step is None  # no Trainer attached, so there is no epoch to report


def test_no_figure_is_rendered_unless_plot_metrics_asks() -> None:
    """The default stays free: the table costs nothing, a figure costs matplotlib per epoch."""

    class _Recorder:
        def __init__(self) -> None:
            self.texts: List[Any] = []
            self.figures: List[Any] = []

        def log_text(self, name: str, text: str, step: Any = None) -> None:
            self.texts.append(name)

        def log_figure(self, name: str, figure: Any, step: Any = None) -> None:
            self.figures.append((name, figure))

    def _report(plot: bool) -> _Recorder:
        tracker = _Recorder()
        trainer = _confusion_trainer()
        trainer.trackers = cast(Any, [tracker])
        trainer.plot_metrics = plot
        trainer.initialize()
        trainer.val_metrics.update(torch.randn(8, 3), torch.randint(0, 3, (8,)))
        trainer.log_dict = lambda payload, **_: None  # type: ignore[method-assign]
        trainer._report_epoch_metrics(trainer.val_metrics)
        return tracker

    off = _report(False)
    assert off.texts == ["val/confusion"] and off.figures == []

    on = _report(True)
    assert on.texts == ["val/confusion"]
    assert len(on.figures) == 1
    name, figure = on.figures[0]
    assert name == "val/confusion" and type(figure).__name__ == "Figure"


def test_one_figure_is_rendered_for_all_trackers() -> None:
    """Rendering is real work; every tracker can be handed the same figure."""

    class _Recorder:
        def __init__(self) -> None:
            self.figures: List[Any] = []

        def log_text(self, name: str, text: str, step: Any = None) -> None:
            pass

        def log_figure(self, name: str, figure: Any, step: Any = None) -> None:
            self.figures.append(figure)

    a, b = _Recorder(), _Recorder()
    trainer = _confusion_trainer()
    trainer.trackers = cast(Any, [a, b])
    trainer.plot_metrics = True
    trainer.initialize()
    trainer.val_metrics.update(torch.randn(8, 3), torch.randint(0, 3, (8,)))
    trainer.log_dict = lambda payload, **_: None  # type: ignore[method-assign]

    trainer._report_epoch_metrics(trainer.val_metrics)

    assert a.figures[0] is b.figures[0]


# ---------------------------------------------------------------------------
# Per-split metric slots: `<split>_metrics` merged over the shared `metrics`
# ---------------------------------------------------------------------------
def _split_trainer(**slots: Any) -> LightningClassifier:
    """Shared `metrics` = accuracy; whatever the caller adds per split."""
    from torchmetrics import Metric
    from torchmetrics.classification import MulticlassAccuracy

    return LightningClassifier(
        model=_TinyClassifier(num_classes=3),
        train_set=_make_records(16),
        num_classes=3,
        balance_classes=False,
        metrics={"accuracy_top1": PartialClass[Metric](MulticlassAccuracy, top_k=1)},
        **slots,
    )


def test_a_split_slot_adds_to_the_shared_set_for_that_split_only() -> None:
    """The point of the split slots: a confusion matrix on validation and nowhere else.

    Merging rather than replacing is what makes that expressible without restating the shared
    metrics beside it — which in a config is duplication that then drifts.
    """
    from torchmetrics import Metric
    from torchmetrics.classification import MulticlassConfusionMatrix

    trainer = _split_trainer(val_metrics={"confusion": PartialClass[Metric](MulticlassConfusionMatrix)})
    trainer.initialize()

    assert sorted(str(k) for k in trainer.val_metrics.keys()) == ["val/accuracy_top1", "val/confusion"]
    assert [str(k) for k in trainer.train_metrics.keys()] == ["train/accuracy_top1"]
    assert [str(k) for k in trainer.test_metrics.keys()] == ["test/accuracy_top1"]


def test_a_split_slot_shadows_a_shared_entry_of_the_same_name() -> None:
    """Merge semantics, so the split can also OVERRIDE — same name, that split's template wins."""
    from torchmetrics import Metric
    from torchmetrics.classification import MulticlassAccuracy

    trainer = _split_trainer(val_metrics={"accuracy_top1": PartialClass[Metric](MulticlassAccuracy, top_k=2)})
    trainer.initialize()

    assert [str(k) for k in trainer.val_metrics.keys()] == ["val/accuracy_top1"]
    assert trainer.val_metrics["val/accuracy_top1"].top_k == 2
    assert trainer.train_metrics["train/accuracy_top1"].top_k == 1


def test_a_ready_made_collection_in_a_split_slot_replaces_instead_of_merging() -> None:
    """There is no key-wise way to merge a built `MetricCollection`, so it takes the split whole.

    Ignoring it would be the worse answer — a wired metric set that silently does nothing.
    """
    from torchmetrics import MetricCollection
    from torchmetrics.classification import MulticlassF1Score

    trainer = _split_trainer(test_metrics=MetricCollection({"f1": MulticlassF1Score(num_classes=3)}))
    trainer.initialize()

    assert [str(k) for k in trainer.test_metrics.keys()] == ["test/f1"]
    assert [str(k) for k in trainer.train_metrics.keys()] == ["train/accuracy_top1"]


def test_the_split_slots_are_built_once() -> None:
    """`initialize()` is called at the top of every `run()`, and the slots are dual-purpose.

    Once a slot may arrive holding a SPEC, `is None` can no longer tell "not built" from
    "wired" — so a second initialize must not re-prefix an already-built collection into
    `val/val/confusion`.
    """
    from torchmetrics import Metric
    from torchmetrics.classification import MulticlassConfusionMatrix

    trainer = _split_trainer(val_metrics={"confusion": PartialClass[Metric](MulticlassConfusionMatrix)})
    trainer.initialize()
    first = trainer.val_metrics
    trainer.initialize()

    assert trainer.val_metrics is first
    assert sorted(str(k) for k in trainer.val_metrics.keys()) == ["val/accuracy_top1", "val/confusion"]


def test_training_step_only_updates_and_the_epoch_end_reports() -> None:
    """The train split reduces ONCE per epoch, exactly as the eval splits do.

    It used to log the per-BATCH values with ``on_step=True``, which lost the training metrics
    in two different ways — Lightning's ``log_every_n_steps`` (50) DROPS a per-step series
    shorter than the interval, so a 32-step run recorded none at all; and what survived a longer
    run was one batch's value where every other backend reports an epoch aggregate. Both were
    measured on the backend sweep before this changed.

    The scalar filter is still applied at the epoch end (``_report_epoch_metrics``), because
    ``log_dict`` carries scalars only — the confusion matrix in this fixture is what would kill
    the run otherwise.
    """
    logged: List[Any] = []
    trainer = _confusion_trainer()
    trainer.initialize()
    trainer.log_dict = lambda payload, **_: logged.append(payload)  # type: ignore[method-assign]
    trainer.log = lambda *a, **k: None  # type: ignore[method-assign]

    trainer.training_step(_make_batch(8, num_classes=3), 0)
    assert logged == [], "the training step must accumulate, not report"

    trainer.on_train_epoch_end()
    assert list(logged[0]) == ["train/accuracy_top1"]


def test_the_train_metrics_reset_at_epoch_START_so_the_last_epoch_survives() -> None:
    """Same rule the eval splits follow, and for the same reason.

    ``_write_metrics_sidecar`` and the ``computed_metrics`` @output both ``compute()`` after
    ``trainer.fit()`` has returned, so resetting at epoch END would hand them an emptied metric.
    """
    trainer = _confusion_trainer()
    trainer.initialize()
    trainer.log_dict = lambda *a, **k: None  # type: ignore[method-assign]
    trainer.log = lambda *a, **k: None  # type: ignore[method-assign]

    batch = _make_batch(8, num_classes=3)
    trainer.training_step(batch, 0)
    accuracy = trainer.train_metrics["train/accuracy_top1"]
    during_epoch = float(accuracy.compute())

    trainer.on_train_epoch_end()
    assert float(accuracy.compute()) == pytest.approx(during_epoch), "the epoch END must not reset"

    trainer.on_train_epoch_start()
    assert accuracy.update_count == 0, "the epoch START is what resets"


# ---------------------------------------------------------------------------
# I/O contract: trained_model @output + Mandatory model/train_set inputs
# ---------------------------------------------------------------------------
def test_declares_trained_model_output() -> None:
    from confluid import output_specs

    specs = {s["name"]: s for s in output_specs(LightningClassifier)}
    assert "trained_model" in specs
    assert specs["trained_model"]["type"] == "Module"
    # The marker rides on the getter so StreamStudio / form-spec can read it. Inherited from
    # the torch family base since 2026-08-06, so the lookup walks the MRO rather than __dict__.
    fget = next(k.__dict__["trained_model"].fget for k in LightningClassifier.__mro__ if "trained_model" in k.__dict__)
    assert getattr(fget, "__confluid_output__", False) is True


def test_inherits_computed_metrics_and_label_names_outputs() -> None:
    """The base matrainer outputs surface for the classification trainer via output_specs' MRO walk."""
    from confluid import output_specs

    names = {s["name"] for s in output_specs(LightningClassifier)}
    assert {"trained_model", "computed_metrics", "class_names"} <= names


def test_trained_model_is_not_a_config_field() -> None:
    from confluid import to_pydantic

    assert "trained_model" not in to_pydantic(LightningClassifier).model_fields


def test_trained_model_raises_before_run() -> None:
    # A lazy (!lazy:) model stays a deferred Fluid until run() flows it — the output guards this.
    trainer = LightningClassifier(
        model=PartialClass(_TinyClassifier, num_classes=3),
        train_set=_make_records(8, num_classes=3),
    )
    with pytest.raises(RuntimeError, match="only available after run"):
        _ = trainer.trained_model


def test_trained_model_returns_live_model_after_initialize() -> None:
    model = _TinyClassifier(num_classes=3)
    trainer = LightningClassifier(
        model=model,
        train_set=_make_records(16, num_classes=3),
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


# ---------------------------------------------------------------------------
# run() dispatch — the @entrypoint markers ARE the table (recordstream run_entrypoint)
# ---------------------------------------------------------------------------


def test_entrypoint_markers_cover_every_task_literal_value() -> None:
    """`run()` dispatches through the markers, so a task the Literal offers must be declared."""
    declared = {str(meta["task"]) for meta in runnable_entrypoints(LightningClassifier).values()}
    assert declared == set(get_args(ClassifierTask))


@pytest.mark.parametrize("task", get_args(ClassifierTask))
def test_run_dispatches_to_the_entry_point_declaring_the_task(task: str) -> None:
    calls: List[str] = []
    # A non-empty stand-in: this test drives DISPATCH only, but run() guards the
    # Mandatory model slot first (matrainer TrainerRunnable._require_model).
    trainer = LightningClassifier(model=cast(Any, object()), task=cast(Any, task))
    trainer.initialize = lambda: None  # type: ignore[method-assign]
    for name in get_args(ClassifierTask):  # instance attrs win over the class's bound methods
        setattr(trainer, name, lambda n=name: calls.append(n))
    trainer.run()
    assert calls == [task]


# --------------------------------------------------------------------------- #
# tracking: one derivation, shared with the torch backend
# --------------------------------------------------------------------------- #


def test_the_tracking_callback_gets_the_runnable_not_a_params_dict() -> None:
    """Both backends hand matrainer the OBJECT, so neither transcribes its own params.

    This class used to walk `get_hierarchy_from_instance` in a private `_config_params` while
    the bare-torch backend hand-wrote six keys — two answers to the same question. The
    derivation now lives in matrainer and both pass `runnable=self`.
    """
    from matrainer import CsvTracker
    from matrainer.lightning import TrackingCallback

    trainer = LightningClassifier(model=nn.Identity(), max_epochs=1)
    trainer.trackers = [CsvTracker()]
    built = trainer._build_trainer()

    # `Trainer.callbacks` is untyped in pytorch_lightning's stubs; the production code
    # appends through the same attribute with the same ignore.
    callbacks = [cb for cb in built.callbacks if isinstance(cb, TrackingCallback)]  # type: ignore[attr-defined]
    assert len(callbacks) == 1
    assert callbacks[0].runnable is trainer
    assert callbacks[0].params is None  # nothing transcribed by hand


def test_no_local_hyperparameter_derivation_survives() -> None:
    """A private re-derivation here would silently diverge from matrainer's again."""
    assert not hasattr(LightningClassifier, "_config_params")
