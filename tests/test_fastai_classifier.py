# mypy: disable-error-code="attr-defined"
"""``FastaiClassifier`` — the fastai ``Learner`` backend over the same torch stack.

The interesting claim this file pins is not "fastai works" but **how little is different**: the
model wrapper, the tensors, the collate and the record keys are the Lightning backend's, and
only the loop changes. So several tests here assert equality with the reference backend rather
than asserting a fastai-specific value.

Skipped when fastai is not installed — it is a selectable backend (``sonair[fastai]``), not a
core dependency.
"""

from typing import Any, List, Tuple, cast

import numpy as np
import pytest

pytest.importorskip("fastai", reason="the fastai backend needs sonair[fastai]")

import torch  # noqa: E402 - after the skip, so a fastai-less install never imports it for this file
from confluid import Fluid, LazyClass, flow, lazy_param_names  # noqa: E402
from matrainer.fastai import as_opt_func as _as_opt_func  # noqa: E402
from matrainer.fastai import named_values as _named_values  # noqa: E402
from matrainer.metric_names import MetricNames  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from sonair.fastai_classifier import FastaiClassifier  # noqa: E402


def _as_metric(name: str) -> Any:
    """Resolve a metric the way a built classifier does — the base's alias-first resolution."""
    return FastaiClassifier(model=torch.nn.Identity())._resolve_metric(name)


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


#: What a test wires when it never touches the model slot. `model` is `Mandatory[Lazy[nn.Module]]`
#: — it must be wired, and it must be a module — so these tests wire the cheapest legal thing
#: rather than `None`, which only type-checked while the slot was `Lazy[Any]`.
_NO_MODEL = torch.nn.Identity()


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
    clf = FastaiClassifier(model=_NO_MODEL)
    assert clf.task == "fit"
    assert clf.num_classes is None
    assert clf.loss is None and clf.metrics is None, "task-shaped slots are sentinels, not live defaults"
    assert clf._learner is None


def test_only_fit_needs_autograd() -> None:
    """A GUI executor evaluates nodes under ``torch.inference_mode()`` and reads this flag."""
    assert FastaiClassifier(model=_NO_MODEL, task="fit").__needs_autograd__ is True
    assert FastaiClassifier(model=_NO_MODEL, task="predict").__needs_autograd__ is False


# ---------------------------------------------------------------------------
# The batch shape — the only thing this backend has to say about classification
# ---------------------------------------------------------------------------


def test_to_tuple_produces_the_pair_fastai_unpacks() -> None:
    """fastai unpacks a batch positionally, so the batch has to BE a tuple — not the record
    dict a ``LightningModule`` reads by key. That is the whole adaptation."""
    clf = FastaiClassifier(model=_NO_MODEL, num_classes=3)
    x, y = clf._to_tuple(_records(4))

    assert x.shape == (4, 3, 8, 8)
    assert y.shape == (4,)
    assert y.dtype is torch.int64, "int64 explicitly — the cross-entropy family rejects int32"


def test_to_tuple_without_targets_is_a_one_tuple() -> None:
    clf = FastaiClassifier(model=_NO_MODEL, num_classes=3)
    assert len(clf._to_tuple(_records(4), with_targets=False)) == 1


def test_a_multilabel_target_becomes_a_float_multi_hot() -> None:
    clf = FastaiClassifier(model=_NO_MODEL, num_classes=3)
    clf.multilabel = True
    _, y = clf._to_tuple(_multilabel_records(4))

    assert y.shape == (4, 3)
    assert y.dtype is torch.float32
    assert y.sum().item() == 8.0, "two labels per record"


def test_a_multilabel_target_without_num_classes_is_a_clear_error() -> None:
    clf = FastaiClassifier(model=_NO_MODEL)
    clf.multilabel = True
    with pytest.raises(RuntimeError, match="num_classes"):
        clf._to_tuple(_multilabel_records(2))


def test_the_batch_shape_matches_the_lightning_backend() -> None:
    """The reuse claim, made checkable: both backends read the same collated record the same way."""
    from recordstream import batch_tensor, collate_records

    records = _records(4)
    clf = FastaiClassifier(model=_NO_MODEL, num_classes=3)
    x, y = clf._to_tuple(records)

    batch = collate_records(records)
    assert torch.equal(x, batch_tensor(batch, "image"))
    assert torch.equal(y, batch_tensor(batch, "class", dtype=torch.int64))


# ---------------------------------------------------------------------------
# Loader construction — the SAME deferred slots the Lightning backend declares
# ---------------------------------------------------------------------------
# This backend used to build its loaders inline in a private `_loader()` helper. That made
# `batch_size`/`num_workers` the ONLY thing a config could say about the data path — no
# `pin_memory`, no `drop_last`, no sampler — while the Lightning backend exposed the whole
# loader as a replaceable slot. The difference was an accident of writing order, not a fastai
# constraint: fastai consumes plain torch loaders.


def test_the_loader_slots_match_the_lightning_backends() -> None:
    """Same names, same deferral, so one config idiom drives both backends."""
    from sonair.lightning_classifier import LightningClassifier

    slots = {"train_loader", "val_loader", "test_loader"}
    assert slots <= lazy_param_names(FastaiClassifier)
    assert slots <= lazy_param_names(LightningClassifier)

    clf = FastaiClassifier(model=_NO_MODEL, batch_size=8, num_workers=2)
    for name in slots:
        stub = getattr(clf, name)
        assert isinstance(stub, Fluid), f"{name} must stay deferred until run() supplies dataset="
        assert stub.kwargs["batch_size"] == 8
        assert stub.kwargs["persistent_workers"] is True  # derived from num_workers, as in Lightning


def test_the_only_divergence_from_lightning_is_the_collate() -> None:
    """fastai unpacks a batch POSITIONALLY, so `_to_tuple` sits where `collate_records` sits."""
    clf = FastaiClassifier(model=_NO_MODEL, num_classes=3)
    train, val = cast(Fluid, clf.train_loader), cast(Fluid, clf.val_loader)

    assert train.kwargs["collate_fn"] == clf._to_tuple
    assert train.kwargs["shuffle"] is True
    assert val.kwargs["shuffle"] is False


def test_a_wired_loader_replaces_the_baked_stub() -> None:
    """What the alignment buys: the loader is a post-construction slot, as in Lightning."""
    clf = FastaiClassifier(model=_NO_MODEL, num_classes=3)
    clf.train_loader = LazyClass(DataLoader, batch_size=1, drop_last=True, collate_fn=clf._to_tuple)

    loader = flow(clf.train_loader, dataset=_records(4))

    assert loader.batch_size == 1 and loader.drop_last is True


def test_predict_gets_record_batches_from_the_SAME_slot() -> None:
    """The predict path overrides `collate_fn` at flow time rather than owning a second loader.

    The sink needs the metadata `_to_tuple` drops, so predict wants record dicts — but every
    other knob the operator set on `test_loader` must still apply.
    """
    from recordstream import collate_records

    clf = FastaiClassifier(model=_NO_MODEL, num_classes=3, batch_size=2)
    records = _records(4)

    tuples = flow(clf.test_loader, dataset=records)
    dicts = flow(clf.test_loader, dataset=records, collate_fn=collate_records)

    assert isinstance(next(iter(tuples)), tuple)
    assert isinstance(next(iter(dicts)), dict), "predict must see the record dict, not the tuple"
    assert dicts.batch_size == 2, "the override must not discard the slot's other kwargs"
    assert flow(clf.test_loader, dataset=records).collate_fn == clf._to_tuple, "stub not mutated"


# ---------------------------------------------------------------------------
# fastai's OWN two objects are deferred slots too
# ---------------------------------------------------------------------------
# `DataLoaders` and `Learner` were built inline until 2026-08-02, which left every knob this
# class does not restate unreachable from config. `DataLoaders(*loaders, …)` takes its loaders
# POSITIONALLY, so nothing a marker can carry could deliver them — they arrive as `flow()`'s
# positional runtime args, the same channel `dataset=` uses for a loader.


def test_the_fastai_objects_are_deferred_slots() -> None:
    assert {"dataloaders", "learner"} <= lazy_param_names(FastaiClassifier)

    clf = FastaiClassifier(model=_NO_MODEL, learning_rate=0.02, weight_decay=0.01)

    assert isinstance(clf.dataloaders, Fluid) and isinstance(clf.learner, Fluid)
    assert cast(Fluid, clf.learner).kwargs == {"lr": 0.02, "wd": 0.01}


def test_a_wired_dataloaders_knob_reaches_the_built_object() -> None:
    """`path` is unreachable any other way — this class does not (and should not) restate it.

    It is what a `Learner` falls back to for its artifact directory when no explicit `path` is
    given, so it decides where `SaveModelCallback` and friends write.
    """
    from fastai.data.core import DataLoaders

    clf = FastaiClassifier(model=_NO_MODEL, num_classes=3, train_set=_records(4), batch_size=2)
    clf.dataloaders = LazyClass(DataLoaders, path="artifacts")

    built = clf._dataloaders()

    assert isinstance(built, DataLoaders)
    assert str(built.path) == "artifacts"
    assert len(built.loaders) == 2, "the two loaders still arrive positionally"


def test_a_wired_learner_knob_reaches_the_built_object() -> None:
    """`moms` / `model_dir` / `wd_bn_bias` / `train_bn` / `splitter` — none is a sonair knob."""
    from fastai.learner import Learner

    clf = FastaiClassifier(model=_Tiny(), num_classes=3, train_set=_records(4), batch_size=2)
    clf._resolve_task_defaults()  # what `initialize()` does: fill the loss sentinel
    clf.learner = LazyClass(Learner, lr=0.05, model_dir="weights", train_bn=False)

    learner = clf._build_learner(clf._dataloaders())

    assert learner.lr == 0.05
    assert str(learner.model_dir) == "weights"
    assert learner.train_bn is False


def test_the_run_supplied_arguments_still_win_over_the_slot() -> None:
    """A config cannot pin the data or the model — those are the run's, and they are kwargs."""
    from fastai.learner import Learner

    model = _Tiny()
    clf = FastaiClassifier(model=model, num_classes=3, train_set=_records(4), batch_size=2)
    clf._resolve_task_defaults()
    clf.learner = LazyClass(Learner, model=_Tiny(num_classes=99))

    learner = clf._build_learner(clf._dataloaders())

    assert learner.model is model


# ---------------------------------------------------------------------------
# Task-shaped defaults
# ---------------------------------------------------------------------------


def test_defaults_are_resolved_from_the_data_not_the_signature() -> None:
    from fastai.losses import BCEWithLogitsLossFlat, CrossEntropyLossFlat

    single = FastaiClassifier(model=_NO_MODEL)
    single._resolve_task_defaults()
    assert isinstance(single.loss, CrossEntropyLossFlat)
    assert single.metrics == ["accuracy"]

    multi = FastaiClassifier(model=_NO_MODEL)
    multi.multilabel = True
    multi._resolve_task_defaults()
    assert isinstance(multi.loss, BCEWithLogitsLossFlat)
    assert multi.metrics == ["accuracy_multi"]


def test_an_explicitly_wired_loss_is_never_replaced() -> None:
    """A live default in the signature would make "chose cross-entropy" and "said nothing"
    indistinguishable — which is why the slot is a ``None`` sentinel."""
    from fastai.losses import LabelSmoothingCrossEntropyFlat

    chosen = LabelSmoothingCrossEntropyFlat()
    clf = FastaiClassifier(model=_NO_MODEL, loss=chosen)
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


class _Recorder:
    metric_names = ["epoch", "train_loss", "valid_loss", "accuracy", "time"]


class _RecordingLearner:
    recorder = _Recorder()


def test_recorder_values_are_matched_to_names_from_the_END() -> None:
    """``validate()`` returns one fewer value than a training epoch (no ``train_loss``), so
    matching from the front would label the validation loss "train_loss"."""
    learner = _RecordingLearner()

    assert _named_values(learner, [0.5, 0.4, 0.9], MetricNames()) == {
        "train/loss": 0.5,
        "val/loss": 0.4,
        "val/accuracy": 0.9,
    }
    assert _named_values(learner, [0.4, 0.9], MetricNames()) == {"val/loss": 0.4, "val/accuracy": 0.9}


def test_an_unprefixed_recorder_column_is_a_VALIDATION_metric() -> None:
    """fastai computes its metrics on the validation pass and prefixes none of them.

    Read as a training metric — which is what a bare key means under every other engine — a
    sweep would report this backend's val accuracy in the train column.
    """
    named = _named_values(_RecordingLearner(), [0.5, 0.4, 0.9], MetricNames())

    assert "val/accuracy" in named and "train/accuracy" not in named


def test_the_names_the_run_records_can_be_moved_to_ANOTHER_engines_spelling() -> None:
    """The `metric_names:` knob, exercised through the function that applies it here."""
    named = _named_values(_RecordingLearner(), [0.5, 0.4, 0.9], MetricNames(target="keras"))

    assert named == {"loss": 0.5, "val_loss": 0.4, "val_accuracy": 0.9}


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

    assert {"train/loss", "val/loss", "val/accuracy"} <= set(clf.computed_metrics)
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

    assert "train/loss" in clf.computed_metrics


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


def test_evaluate_scores_the_test_set() -> None:
    clf = FastaiClassifier(model=_Tiny(), test_set=_stream(_records(8)), task="test", batch_size=4)
    clf.run()

    assert "val/loss" in clf.computed_metrics


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
    """Raised at Learner assembly (`_learner_model`), which `run()` reaches right after the
    task's `initialize()` — still during SETUP, before any batch is trained on.

    The fit path is what this covers, because it is the only one that reaches the check: an
    eval task resolves `num_classes` OFF the model first (`_initialize_eval`) and so rejects a
    non-model earlier, with its own equally clear message.
    """
    clf = FastaiClassifier(model=cast(Any, "not a model"), train_set=_stream(_records(4)), num_classes=3)
    with pytest.raises(TypeError, match="nn.Module"):
        clf.run()


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


# --------------------------------------------------------------------------- #
# Metrics: the `f1_macro` alias, and the training columns
# --------------------------------------------------------------------------- #
def test_f1_macro_names_a_CONFIGURED_F1Score_not_the_factory() -> None:
    """`fastai.metrics` publishes `F1Score` as a FACTORY, so a bare name resolves to the wrong
    thing: `getattr(fastai.metrics, "F1Score")` hands the Learner the factory itself, which is
    accepted without complaint and then reported under fastai's own `f1_score` column with the
    default `average="binary"` — which on a multi-class problem raises inside scikit-learn.

    The alias resolves to a BUILT metric, renamed to the column every other backend writes.
    """
    metric = _as_metric("f1_macro")

    assert metric.name == "f1_macro", "the column a cross-framework comparison lines up on"
    assert not isinstance(metric, type) and callable(metric)
    assert _as_metric("f1_micro").name == "f1_micro"
    assert _as_metric("f1_weighted").name == "f1_weighted"


def test_the_unknown_metric_error_names_the_aliases_too() -> None:
    """`f1_macro` is not a `fastai.metrics` attribute, so the message has to say where it IS."""
    with pytest.raises(ValueError, match="f1_macro"):
        _as_metric("no_such_metric")


def test_a_fit_reports_the_metrics_on_the_TRAINING_pass_too() -> None:
    """`Recorder.train_metrics` — accumulated over the training batches, not a second pass.

    The old claim was that a `Learner` "computes its metrics on the validation pass only". That
    is its DEFAULT, not a constraint: `Recorder.__init__` takes `train_metrics=False`, and with
    it on the columns become `train_loss, train_<m>…, valid_loss, valid_<m>…` — exactly the
    shape `matrainer.metric_names`' fastai vocabulary already reads, which is why turning it on
    needed no translation change.
    """
    clf = FastaiClassifier(
        model=_Tiny(),
        train_set=_stream(_records(12)),
        val_set=_stream(_records(8)),
        num_classes=3,
        max_epochs=1,
        batch_size=4,
        metrics=["accuracy", "f1_macro"],
    )
    clf.run()

    assert {"train/accuracy", "train/f1_macro"} <= set(clf.computed_metrics)
    assert {"val/accuracy", "val/f1_macro"} <= set(clf.computed_metrics)
    assert clf._learner is not None and clf._learner.recorder.train_metrics is True


def test_the_fit_path_UNIONS_train_and_val_while_the_test_path_narrows() -> None:
    """A fitting `Learner` reports ONE list on both passes, so the two cannot be told apart.

    `Recorder._train_mets` and `_valid_mets` are the same `self.metrics` with different
    prefixes, so a val-only metric IS computed on training as well — the same widening
    `keras.Model.compile()` forces, and the same trade: noise a reader skips beats a wired
    metric that silently reports nothing. The TEST path builds its own Learner and runs one
    pass, so it keeps the narrowing merge.
    """
    clf = FastaiClassifier(
        model=_NO_MODEL,
        num_classes=3,
        metrics=["accuracy"],
        train_metrics=["error_rate"],
        val_metrics=["f1_macro"],
        test_metrics=["top_k_accuracy"],
    )

    assert set(clf._learner_metrics("val")) == {"accuracy", "error_rate", "f1_macro"}
    assert set(clf._learner_metrics("test")) == {"accuracy", "top_k_accuracy"}
