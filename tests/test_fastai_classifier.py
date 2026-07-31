# mypy: disable-error-code="attr-defined"
"""``FastaiClassifier`` — the fastai ``Learner`` backend over the same torch stack.

The interesting claim this file pins is not "fastai works" but **how little is different**: the
model wrapper, the tensors, the collate and the record keys are the Lightning backend's, and
only the loop changes. So several tests here assert equality with the reference backend rather
than asserting a fastai-specific value.

Skipped when fastai is not installed — it is a selectable backend (``sonair[fastai]``), not a
core dependency.
"""

from pathlib import Path
from typing import Any, List, Tuple, cast

import numpy as np
import pytest

pytest.importorskip("fastai", reason="the fastai backend needs sonair[fastai]")

import torch  # noqa: E402 - after the skip, so a fastai-less install never imports it for this file

from sonair.fastai_classifier import FastaiClassifier, _as_metric, _as_opt_func, _named_values  # noqa: E402


class _CaptureSink:
    def __init__(self) -> None:
        self.rows: List[Tuple[Any, Any]] = []
        self.closed = False

    def write(self, prediction: Any, metadata: Any) -> None:
        self.rows.append((prediction, metadata))

    def close(self) -> None:
        self.closed = True


class _Tiny(torch.nn.Module):
    """A model with the one attribute the eval path reads off a concrete model: ``num_classes``."""

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.fc = torch.nn.Linear(3 * 8 * 8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.fc(x.flatten(1)))


def _records(n: int, num_classes: int = 3) -> List[dict]:
    """Channels-FIRST records — what a torch preprocessing chain (``ToTensor``) produces."""
    from recordstream.items import Image

    rng = np.random.default_rng(0)
    return [
        {
            "image": Image(torch.as_tensor(rng.random((3, 8, 8), dtype=np.float32))),
            "class": i % num_classes,
            "row_id": i,
        }
        for i in range(n)
    ]


def _multilabel_records(n: int, num_classes: int = 3) -> List[dict]:
    from recordstream import MultiLabel
    from recordstream.items import Image

    rng = np.random.default_rng(1)
    return [
        {
            "image": Image(torch.as_tensor(rng.random((3, 8, 8), dtype=np.float32))),
            "class": MultiLabel([i % num_classes, (i + 1) % num_classes]),
            "row_id": i,
        }
        for i in range(n)
    ]


def _stream(records: List[dict]) -> Any:
    from recordstream import Stream

    return Stream(source=records)


# ---------------------------------------------------------------------------
# Discovery + construction
# ---------------------------------------------------------------------------


def test_discovery_tags() -> None:
    """``framework="torch"`` because that is what it consumes — fastai is a layer on torch, and
    its losses are torch callables a Lightning run can take as well."""
    assert FastaiClassifier.__confluid_task__ == "classification"
    assert FastaiClassifier.__confluid_role__ == "trainer"
    assert FastaiClassifier.__confluid_framework__ == "torch"


def test_zero_arg_construction_does_no_work() -> None:
    """The workspace lazy-init mandate: the constructor stores, it does not build."""
    clf = FastaiClassifier(model=None)
    assert clf.task == "fit"
    assert clf.num_classes is None
    assert clf.loss is None and clf.metrics is None, "task-shaped slots are sentinels, not live defaults"
    assert clf._learner is None


def test_only_fit_needs_autograd() -> None:
    """A GUI executor evaluates nodes under ``torch.inference_mode()`` and reads this flag."""
    assert FastaiClassifier(model=None, task="fit").__needs_autograd__ is True
    assert FastaiClassifier(model=None, task="predict").__needs_autograd__ is False


# ---------------------------------------------------------------------------
# The batch shape — the only thing this backend has to say about classification
# ---------------------------------------------------------------------------


def test_to_tuple_produces_the_pair_fastai_unpacks() -> None:
    """fastai unpacks a batch positionally, so the batch has to BE a tuple — not the record
    dict a ``LightningModule`` reads by key. That is the whole adaptation."""
    clf = FastaiClassifier(model=None, num_classes=3)
    x, y = clf._to_tuple(_records(4))

    assert x.shape == (4, 3, 8, 8)
    assert y.shape == (4,)
    assert y.dtype is torch.int64, "int64 explicitly — the cross-entropy family rejects int32"


def test_to_tuple_without_targets_is_a_one_tuple() -> None:
    clf = FastaiClassifier(model=None, num_classes=3)
    assert len(clf._to_tuple(_records(4), with_targets=False)) == 1


def test_a_multilabel_target_becomes_a_float_multi_hot() -> None:
    clf = FastaiClassifier(model=None, num_classes=3)
    clf.multilabel = True
    _, y = clf._to_tuple(_multilabel_records(4))

    assert y.shape == (4, 3)
    assert y.dtype is torch.float32
    assert y.sum().item() == 8.0, "two labels per record"


def test_a_multilabel_target_without_num_classes_is_a_clear_error() -> None:
    clf = FastaiClassifier(model=None)
    clf.multilabel = True
    with pytest.raises(RuntimeError, match="num_classes"):
        clf._to_tuple(_multilabel_records(2))


def test_the_batch_shape_matches_the_lightning_backend() -> None:
    """The reuse claim, made checkable: both backends read the same collated record the same way."""
    from recordstream import batch_tensor, collate_records

    records = _records(4)
    clf = FastaiClassifier(model=None, num_classes=3)
    x, y = clf._to_tuple(records)

    batch = collate_records(records)
    assert torch.equal(x, batch_tensor(batch, "image"))
    assert torch.equal(y, batch_tensor(batch, "class", dtype=torch.int64))


# ---------------------------------------------------------------------------
# Task-shaped defaults
# ---------------------------------------------------------------------------


def test_defaults_are_resolved_from_the_data_not_the_signature() -> None:
    from fastai.losses import BCEWithLogitsLossFlat, CrossEntropyLossFlat

    single = FastaiClassifier(model=None)
    single._resolve_task_defaults()
    assert isinstance(single.loss, CrossEntropyLossFlat)
    assert single.metrics == ["accuracy"]

    multi = FastaiClassifier(model=None)
    multi.multilabel = True
    multi._resolve_task_defaults()
    assert isinstance(multi.loss, BCEWithLogitsLossFlat)
    assert multi.metrics == ["accuracy_multi"]


def test_an_explicitly_wired_loss_is_never_replaced() -> None:
    """A live default in the signature would make "chose cross-entropy" and "said nothing"
    indistinguishable — which is why the slot is a ``None`` sentinel."""
    from fastai.losses import LabelSmoothingCrossEntropyFlat

    chosen = LabelSmoothingCrossEntropyFlat()
    clf = FastaiClassifier(model=None, loss=chosen)
    clf._resolve_task_defaults()
    clf._resolve_task_defaults()  # idempotent

    assert clf.loss is chosen


# ---------------------------------------------------------------------------
# The string resolvers — fastai's metrics are FUNCTIONS, so a name is all YAML can write
# ---------------------------------------------------------------------------


def test_a_metric_name_resolves_against_fastai_metrics() -> None:
    from fastai.metrics import accuracy, top_k_accuracy

    assert _as_metric("accuracy") is accuracy
    assert _as_metric("top_k_accuracy") is top_k_accuracy


def test_a_metric_callable_passes_through() -> None:
    from fastai.metrics import error_rate

    assert _as_metric(error_rate) is error_rate


def test_an_unknown_metric_name_says_where_names_come_from() -> None:
    with pytest.raises(ValueError, match="fastai.metrics"):
        _as_metric("definitely_not_a_metric")


def test_an_optimizer_name_resolves_against_fastai_optimizer() -> None:
    from fastai.optimizer import SGD, Adam

    assert _as_opt_func("Adam") is Adam
    assert _as_opt_func("SGD") is SGD


def test_an_unknown_optimizer_name_says_where_names_come_from() -> None:
    with pytest.raises(ValueError, match="fastai.optimizer"):
        _as_opt_func("definitely_not_an_optimizer")


# ---------------------------------------------------------------------------
# Recorder bookkeeping
# ---------------------------------------------------------------------------


def test_recorder_values_are_matched_to_names_from_the_END() -> None:
    """``validate()`` returns one fewer value than a training epoch (no ``train_loss``), so
    matching from the front would label the validation loss "train_loss"."""

    class _Recorder:
        metric_names = ["epoch", "train_loss", "valid_loss", "accuracy", "time"]

    class _Learner:
        recorder = _Recorder()

    learner = _Learner()
    assert _named_values(learner, [0.5, 0.4, 0.9]) == {"train_loss": 0.5, "valid_loss": 0.4, "accuracy": 0.9}
    assert _named_values(learner, [0.4, 0.9]) == {"valid_loss": 0.4, "accuracy": 0.9}


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def test_fit_trains_and_reports_the_last_epochs_metrics() -> None:
    clf = FastaiClassifier(
        model=_Tiny(),
        train_set=_stream(_records(12)),
        val_set=_stream(_records(8)),
        num_classes=3,
        max_epochs=2,
        batch_size=4,
    )
    clf.run()

    assert {"train_loss", "valid_loss", "accuracy"} <= set(clf.computed_metrics)
    assert clf._learner is not None and len(clf._learner.recorder.values) == 2


def test_fit_without_one_cycle_uses_the_flat_schedule() -> None:
    clf = FastaiClassifier(
        model=_Tiny(),
        train_set=_stream(_records(8)),
        val_set=_stream(_records(8)),
        num_classes=3,
        max_epochs=1,
        batch_size=4,
        one_cycle=False,
    )
    clf.run()

    assert "train_loss" in clf.computed_metrics


def test_no_val_set_falls_back_to_the_training_data() -> None:
    """fastai's ``Learner`` ALWAYS validates — it has no "skip the val loop" mode the way
    ``pytorch_lightning.Trainer`` does — so something has to fill ``dls.valid``. The training
    data does, which keeps a run working and makes the reported metrics measure fit; the
    classifier logs a warning saying exactly that."""
    clf = FastaiClassifier(model=_Tiny(), train_set=_stream(_records(8)), num_classes=3, max_epochs=1, batch_size=4)
    clf.initialize()
    dataloaders = clf._dataloaders()

    assert dataloaders.valid.dataset is dataloaders.train.dataset
    assert dataloaders.valid.sampler.__class__ is not dataloaders.train.sampler.__class__, "val must not shuffle"


def test_a_wired_val_set_is_used_for_validation() -> None:
    train, val = _stream(_records(8)), _stream(_records(4))
    clf = FastaiClassifier(model=_Tiny(), train_set=train, val_set=val, num_classes=3, batch_size=4)
    clf.initialize()
    dataloaders = clf._dataloaders()

    assert dataloaders.valid.dataset is not dataloaders.train.dataset
    assert len(dataloaders.valid.dataset) == 4


def test_a_wired_callback_gets_its_parent_directory_created(tmp_path: Path) -> None:
    """fastai's ``CSVLogger`` opens ``learn.path/fname`` without creating it — the same failure
    the Keras backend guards, and it lands minutes into a run rather than at the start."""
    from fastai.callback.progress import CSVLogger

    clf = FastaiClassifier(
        model=_Tiny(),
        train_set=_stream(_records(8)),
        val_set=_stream(_records(8)),
        num_classes=3,
        max_epochs=1,
        batch_size=4,
        logger=[CSVLogger(fname="not_yet/history.csv")],
    )
    clf.metrics_sidecar_path = str(tmp_path / "metrics.json")
    clf.run()

    rows = (tmp_path / "not_yet" / "history.csv").read_text().strip().splitlines()
    assert rows[0].startswith("epoch,")
    assert len(rows) == 2, "a header plus one row per epoch"


def test_evaluate_scores_the_test_set() -> None:
    clf = FastaiClassifier(model=_Tiny(), test_set=_stream(_records(8)), task="test", batch_size=4)
    clf.run()

    assert "valid_loss" in clf.computed_metrics


def test_evaluate_reads_num_classes_off_the_LIVE_model() -> None:
    """The trained head is the single source of truth — an eval split may be a subset."""
    clf = FastaiClassifier(model=_Tiny(num_classes=5), test_set=_stream(_records(8, num_classes=3)), task="test")
    clf.initialize()

    assert clf.num_classes == 5


def test_a_model_without_num_classes_is_a_clear_error_on_the_eval_path() -> None:
    clf = FastaiClassifier(model=torch.nn.Linear(2, 2), task="test")
    with pytest.raises(RuntimeError, match="num_classes"):
        clf.initialize()


def test_predict_writes_one_entry_per_record_with_its_metadata() -> None:
    sink = _CaptureSink()
    clf = FastaiClassifier(model=_Tiny(), test_set=_stream(_records(7)), task="predict", batch_size=3)
    clf.predictions_sink = sink  # type: ignore[assignment]
    clf.run()

    assert len(sink.rows) == 7, "a short final batch must not be dropped"
    prediction, metadata = sink.rows[0]
    assert set(prediction) == {"logits", "probs", "class_idx"}, "the contract, not raw logits"
    assert metadata == {"row_id": 0}
    assert pytest.approx(float(prediction["probs"].sum()), abs=1e-5) == 1.0
    assert sink.closed


def test_num_classes_is_derived_from_the_data_when_not_given() -> None:
    clf = FastaiClassifier(model=_Tiny(), train_set=_stream(_records(9, num_classes=3)), max_epochs=1)
    clf.initialize()

    assert clf.num_classes == 3


def test_string_targets_are_encoded_through_a_fitted_label_map() -> None:
    from recordstream.items import Image

    records = [
        {"image": Image(torch.zeros(3, 8, 8)), "class": name} for name in ["cat", "dog", "cat", "bird", "dog", "bird"]
    ]
    clf = FastaiClassifier(model=_Tiny(), train_set=_stream(records), max_epochs=1)
    clf.initialize()

    assert clf.num_classes == 3
    assert clf.class_names == ["bird", "cat", "dog"], "deterministic sorted-unique ordering"


def test_fit_without_a_train_set_is_a_clear_error() -> None:
    clf = FastaiClassifier(model=_Tiny(), num_classes=3)
    with pytest.raises(ValueError, match="train_set"):
        clf.fit()


@pytest.mark.parametrize("method", ["test", "predict"])
def test_eval_verbs_without_a_test_set_are_clear_errors(method: str) -> None:
    clf = FastaiClassifier(model=_Tiny(), num_classes=3)
    with pytest.raises(ValueError, match="test_set"):
        getattr(clf, method)()


def test_a_non_module_model_is_a_clear_error() -> None:
    clf = FastaiClassifier(model="not a model")
    with pytest.raises(TypeError, match="nn.Module"):
        clf._live_model()


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def test_outputs_raise_while_the_model_is_still_deferred() -> None:
    from confluid import LazyClass

    clf = FastaiClassifier(model=LazyClass(_Tiny))
    with pytest.raises(RuntimeError, match="after run"):
        _ = clf.trained_model


def test_trained_model_is_the_live_module_after_a_run() -> None:
    model = _Tiny()
    clf = FastaiClassifier(
        model=model, train_set=_stream(_records(8)), val_set=_stream(_records(8)), num_classes=3, max_epochs=1
    )
    clf.run()

    assert clf.trained_model is model


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
