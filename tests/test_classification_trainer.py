"""Tests for LightningClassificationTrainer."""

from typing import Any

import torch
import torch.nn as nn
from dataflux.sample import Sample
from torchmetrics import MetricCollection

from sonair.trainer import LightningClassificationTrainer, _build_classification_metrics

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


def _make_samples(n: int = 16, num_classes: int = 3, input_size: int = 4) -> list[Sample]:
    return [Sample(input=torch.randn(input_size), target=i % num_classes, metadata={}) for i in range(n)]


# ---------------------------------------------------------------------------
# _build_classification_metrics
# ---------------------------------------------------------------------------


def test_build_metrics_includes_top5_only_when_enough_classes() -> None:
    m2 = _build_classification_metrics(2)
    assert "accuracy_top5" not in m2
    assert "accuracy_top1" in m2
    assert "f1_macro" in m2

    m5 = _build_classification_metrics(5)
    assert "accuracy_top5" in m5


def test_build_metrics_produces_metric_collection() -> None:
    m = _build_classification_metrics(3)
    assert isinstance(m, MetricCollection)


# ---------------------------------------------------------------------------
# LightningClassificationTrainer defaults
# ---------------------------------------------------------------------------


def test_loss_fn_defaults_to_cross_entropy() -> None:
    trainer = LightningClassificationTrainer(
        model=_TinyClassifier(num_classes=3),
        train_set=_make_samples(16),
        num_classes=3,
    )
    assert isinstance(trainer.loss_fn, nn.CrossEntropyLoss)


def test_metrics_are_wired_with_correct_prefixes() -> None:
    trainer = LightningClassificationTrainer(
        model=_TinyClassifier(num_classes=3),
        train_set=_make_samples(16),
        num_classes=3,
    )
    assert trainer.train_metrics is not None
    assert trainer.val_metrics is not None
    assert trainer.test_metrics is not None
    assert isinstance(trainer.train_metrics, MetricCollection)
    assert isinstance(trainer.val_metrics, MetricCollection)
    assert isinstance(trainer.test_metrics, MetricCollection)

    train_keys = [str(k) for k in trainer.train_metrics.keys()]
    val_keys = [str(k) for k in trainer.val_metrics.keys()]
    test_keys = [str(k) for k in trainer.test_metrics.keys()]

    assert all(k.startswith("train/") for k in train_keys), f"Expected train/ prefix, got: {train_keys}"
    assert all(k.startswith("val/") for k in val_keys), f"Expected val/ prefix, got: {val_keys}"
    assert all(k.startswith("test/") for k in test_keys), f"Expected test/ prefix, got: {test_keys}"


def test_no_top5_for_small_num_classes() -> None:
    trainer = LightningClassificationTrainer(
        model=_TinyClassifier(num_classes=2),
        train_set=_make_samples(16, num_classes=2),
        num_classes=2,
    )
    assert trainer.train_metrics is not None
    keys = [str(k) for k in trainer.train_metrics.keys()]
    assert not any("top5" in k for k in keys), f"top5 should not appear for 2 classes, got: {keys}"


def test_top5_added_for_large_num_classes() -> None:
    trainer = LightningClassificationTrainer(
        model=_TinyClassifier(num_classes=10),
        train_set=_make_samples(16, num_classes=10),
        num_classes=10,
    )
    assert trainer.train_metrics is not None
    keys = [str(k) for k in trainer.train_metrics.keys()]
    assert any("top5" in k for k in keys), f"top5 should appear for 10 classes, got: {keys}"


# ---------------------------------------------------------------------------
# Forward / step methods
# ---------------------------------------------------------------------------


def test_training_step_returns_scalar_loss() -> None:
    trainer = LightningClassificationTrainer(
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
    trainer = LightningClassificationTrainer(
        model=_TinyClassifier(num_classes=3),
        train_set=_make_samples(16),
        num_classes=3,
    )
    batch = _make_batch(n=8, num_classes=3)
    loss = trainer.validation_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0


def test_metrics_update_on_training_step() -> None:
    trainer = LightningClassificationTrainer(
        model=_TinyClassifier(num_classes=3),
        train_set=_make_samples(16),
        num_classes=3,
    )
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


class _DimsDataset:
    """Map-style dataset exposing the derived_dimensions() hook (like sonair's)."""

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
    trainer = LightningClassificationTrainer(
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
    from torch.utils.data import DataLoader

    from sonair.classification import classification_collate_fn

    # Minimal constructor; everything else configured post-construction (the paradigm).
    trainer = LightningClassificationTrainer(
        model=_TinyClassifier(num_classes=4),
        train_set=_DimsDataset(16, num_classes=4),
        # num_classes omitted -> derived from the dataset
    )
    trainer.train_loader = ConfluidClass(DataLoader, batch_size=8, collate_fn=classification_collate_fn)
    trainer.lightning = _cpu_lightning(tmp_path)
    trainer.run()
    assert isinstance(trainer.val_metrics, MetricCollection)
    sized = [m for m in trainer.val_metrics.values() if hasattr(m, "num_classes")]
    assert sized and all(m.num_classes == 4 for m in sized)


def test_default_loader_uses_classification_collate_fn() -> None:
    from confluid.fluid import Class as ConfluidClass

    from sonair.classification import classification_collate_fn

    trainer = LightningClassificationTrainer(
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


def test_ergonomic_knobs_fold_into_loader_and_lightning_stubs() -> None:
    from confluid import Class as ConfluidClass

    trainer = LightningClassificationTrainer(
        model=_TinyClassifier(num_classes=3), train_set=_make_samples(16), num_classes=3
    )
    # Knobs are plain attributes (set here as broadcasting would), folded at run().
    trainer.batch_size = 64
    trainer.num_workers = 4
    trainer.persist_workers = True
    trainer.max_epochs = 7
    trainer._apply_loader_knobs()
    trainer._apply_lightning_knobs()

    tl = trainer.train_loader.kwargs
    assert tl["batch_size"] == 64 and tl["num_workers"] == 4
    assert tl["shuffle"] is True and tl["persistent_workers"] is True
    assert trainer.val_loader.kwargs["shuffle"] is False
    assert isinstance(trainer.lightning, ConfluidClass)
    assert trainer.lightning.kwargs["max_epochs"] == 7


def test_persistent_workers_guarded_when_num_workers_zero() -> None:
    # persist_workers=True is meaningless (and raises in torch) when num_workers=0.
    trainer = LightningClassificationTrainer(
        model=_TinyClassifier(num_classes=3), train_set=_make_samples(16), num_classes=3
    )
    trainer.num_workers = 0
    trainer.persist_workers = True
    trainer._apply_loader_knobs()
    assert trainer.train_loader.kwargs["persistent_workers"] is False


def test_explicit_loader_keeps_its_own_batch_size() -> None:
    from confluid import Class as ConfluidClass
    from torch.utils.data import DataLoader

    trainer = LightningClassificationTrainer(
        model=_TinyClassifier(num_classes=3), train_set=_make_samples(16), num_classes=3
    )
    trainer.batch_size = 8
    trainer.train_loader = ConfluidClass(DataLoader, batch_size=999)  # explicit override
    trainer._apply_loader_knobs()  # setdefault must NOT clobber the explicit value
    assert trainer.train_loader.kwargs["batch_size"] == 999


# ---------------------------------------------------------------------------
# Flat-config broadcasting + logger deferred-construction (the user's ask)
# ---------------------------------------------------------------------------


_FLAT_YAML = """
trainer: !class:sonair.LightningClassificationTrainer
  run_name: x
model: "MODEL_SENTINEL"
train_set: "TRAIN_SENTINEL"
val_set: "VAL_SENTINEL"
batch_size: 64
num_workers: 4
persist_workers: true
max_epochs: 7
lightning_loggers:
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
    assert t.train_set == "TRAIN_SENTINEL"
    assert t.val_set == "VAL_SENTINEL"
    # Scalar ergonomic knobs broadcast into the trainer's attributes...
    assert t.batch_size == 64 and t.num_workers == 4 and t.persist_workers is True and t.max_epochs == 7
    assert isinstance(t.lightning_loggers, list) and len(t.lightning_loggers) == 1
    # ...and run()'s fold pushes them into the deferred stubs.
    t._apply_loader_knobs()
    t._apply_lightning_knobs()
    assert t.train_loader.kwargs["batch_size"] == 64
    assert t.train_loader.kwargs["persistent_workers"] is True
    assert t.lightning.kwargs["max_epochs"] == 7


def test_lightning_loggers_broadcast_and_preflowed_into_lightning() -> None:
    """`lightning_loggers: [...]` (List-annotated) broadcasts; its `!class:` Fluids
    are pre-flowed so the deferred L.Trainer gets LIVE logger instances."""
    from confluid import flow, load
    from confluid.fluid import Fluid
    from pytorch_lightning.loggers import TensorBoardLogger

    loaded = load(_FLAT_YAML)
    t = loaded["trainer"]
    t = flow(t) if isinstance(t, Fluid) else t

    # Broadcast in as deferred Fluids...
    assert all(isinstance(x, Fluid) for x in t.lightning_loggers)
    t._apply_lightning_knobs()
    loggers = t.lightning.kwargs["logger"]
    assert isinstance(loggers, list) and len(loggers) == 1
    # ...pre-flowed to a live instance, NOT an unmaterialized Class/Fluid.
    assert isinstance(loggers[0], TensorBoardLogger)
    assert not isinstance(loggers[0], Fluid)


# ---------------------------------------------------------------------------
# !lazy: model -> num_classes injected from the dataset at run time.
# ---------------------------------------------------------------------------


def test_lazy_model_gets_num_classes_injected_at_run(tmp_path: object) -> None:
    from confluid import Class as ConfluidClass
    from confluid import LazyClass
    from torch.utils.data import DataLoader

    from sonair.classification import classification_collate_fn

    # Model declared !lazy: WITHOUT num_classes -> derived from the dataset (4).
    trainer = LightningClassificationTrainer(
        model=LazyClass(_TinyClassifier),  # no num_classes; default 3 would be wrong
        train_set=_DimsDataset(16, num_classes=4),
    )
    trainer.train_loader = ConfluidClass(DataLoader, batch_size=8, collate_fn=classification_collate_fn)
    trainer.lightning = _cpu_lightning(tmp_path)
    trainer.run()
    # The lazy model was built with the derived num_classes=4 (fc out_features).
    assert isinstance(trainer.model, _TinyClassifier)
    assert trainer.model.fc.out_features == 4
