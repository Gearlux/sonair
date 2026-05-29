"""Tests for LightningClassificationTrainer."""

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
        metadata={"per_sample": [{"idx": i} for i in range(n)]},
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


def test_omitted_num_classes_defers_metrics() -> None:
    """With num_classes omitted, metrics are deferred (a Lazy fluid), not eager."""
    from confluid.fluid import Lazy

    trainer = LightningClassificationTrainer(
        model=_TinyClassifier(num_classes=4),
        train_set=_make_samples(16, num_classes=4),
    )
    assert isinstance(trainer.val_metrics, Lazy)
    assert isinstance(trainer.train_metrics, Lazy)


def test_omitted_num_classes_derived_and_built_at_run(tmp_path: object) -> None:
    import pytorch_lightning as L
    from confluid.fluid import Class as ConfluidClass
    from torch.utils.data import DataLoader

    from sonair.classification import classification_collate_fn

    trainer = LightningClassificationTrainer(
        model=_TinyClassifier(num_classes=4),
        train_set=_DimsDataset(16, num_classes=4),
        # num_classes omitted -> derived from the dataset
        optimizer=ConfluidClass(torch.optim.SGD, lr=0.01),
        train_loader=ConfluidClass(DataLoader, batch_size=8, collate_fn=classification_collate_fn),
        lightning=ConfluidClass(
            L.Trainer,
            max_epochs=1,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            limit_train_batches=1,
            default_root_dir=str(tmp_path),
        ),
    )
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
