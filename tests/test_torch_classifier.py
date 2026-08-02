"""``TorchClassifier`` — the hand-written PyTorch loop.

The backend with nothing underneath it, so these tests are unusually literal: they check that
the loop does the things a ``Trainer`` would have done for you, because here nobody does them
unless this class does. Gradient steps actually descend the loss, ``model.eval()`` is entered
for the eval pass, metrics are reset between epochs, and the deferred slots (loaders, optimizer,
loss) really are the ones driving the run.

The rest is parity: this backend and the Lightning one must agree on the label pipeline and the
loss/metric defaults, since the project's claim is that a config moves between them unchanged.
"""

# mypy: disable-error-code="attr-defined"

import inspect
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

pytest.importorskip("torch", reason="needs sonair[torch]")

# Imported normally AFTER the skip so the names are real modules to a type checker; the
# `importorskip` above is what keeps the file collectable on an install without the extra.
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from marainer import CsvTracker  # noqa: E402
from recordstream import Stream  # noqa: E402
from recordstream.items import Image, Label  # noqa: E402

from sonair.torch_classifier import TorchClassifier, _resolve_device  # noqa: E402

NUM_CLASSES = 3


class _Tiny(nn.Module):
    """A model, not a wrapper: the classifier addresses whatever is in its slot directly."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(3 * 8 * 8, num_classes)

    def forward(self, x: Any) -> Any:
        return self.fc(x.flatten(1))


class _CaptureSink:
    def __init__(self) -> None:
        self.rows: List[Tuple[Any, Any]] = []
        self.closed = False

    def write(self, prediction: Any, metadata: Any) -> None:
        self.rows.append((prediction, metadata))

    def close(self) -> None:
        self.closed = True


def _records(n: int, names: bool = False) -> List[Dict[str, Any]]:
    """Channels-FIRST records — the shape a torch preprocessing chain produces."""
    rng = np.random.default_rng(0)
    return [
        {
            "image": Image(rng.random((3, 8, 8)).astype("float32"), layout="CHW"),
            "class": Label("abc"[i % NUM_CLASSES] if names else i % NUM_CLASSES),
            "row_id": i,
        }
        for i in range(n)
    ]


def _clf(**kwargs: Any) -> TorchClassifier:
    kwargs.setdefault("model", _Tiny())
    kwargs.setdefault("batch_size", 4)
    kwargs.setdefault("device", "cpu")
    return TorchClassifier(**kwargs)


# --------------------------------------------------------------------------- #
# It is a plain torch loop — no framework underneath
# --------------------------------------------------------------------------- #


def test_it_needs_no_training_framework() -> None:
    """The claim the backend exists to make, checked rather than trusted.

    A `pytorch_lightning` / `keras` / `fastai` import here would silently turn the control
    experiment into another wrapper — and it would also break `pip install 'sonair[torch]'`,
    whose whole point is that the extra carries no training framework. Read as IMPORT
    STATEMENTS via `ast`, not as substrings: the module docstring names all four frameworks
    while explaining what it does without them, so a text search finds the prose.
    """
    import ast

    tree = ast.parse((Path(__file__).resolve().parents[1] / "sonair" / "torch_classifier.py").read_text())
    imported: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    for framework in ("pytorch_lightning", "keras", "fastai", "transformers"):
        offenders = {m for m in imported if m == framework or m.startswith(f"{framework}.")}
        assert not offenders, f"the bare-torch backend must not import {framework}: {offenders}"
    # marainer's metric helpers are reached at their Lightning-FREE home; the `marainer.lightning`
    # package runs `require("pytorch_lightning")` on import, which would defeat the extra.
    assert "marainer.torch" in imported
    assert not {m for m in imported if m.startswith("marainer.lightning")}


def test_it_declares_the_torch_framework_axis() -> None:
    """Same picker as the Lightning backend: both are wired to `torch.nn` losses and torchmetrics."""
    assert TorchClassifier.__confluid_task__ == "classification"
    assert TorchClassifier.__confluid_role__ == "trainer"
    assert TorchClassifier.__confluid_framework__ == "torch"


def test_only_fit_needs_autograd() -> None:
    """A GUI executor may call `run()` inside `torch.inference_mode()`; only training objects."""
    assert _clf(task="fit").__needs_autograd__ is True
    assert _clf(task="predict").__needs_autograd__ is False


def test_zero_arg_construction_does_no_work() -> None:
    """The workspace lazy-init mandate: the constructor stores, it does not build."""
    clf = _clf()

    assert clf.num_classes is None
    assert clf.loss_fn is None and clf.metrics is None
    assert clf.train_metrics is None


@pytest.mark.parametrize("choice", ["cpu", "auto"])
def test_the_device_is_resolved_not_guessed(choice: str) -> None:
    """`auto` picks what is present; an explicit choice is taken at its word."""
    resolved = _resolve_device(choice)  # type: ignore[arg-type]

    assert isinstance(resolved, torch.device)
    if choice == "cpu":
        assert resolved.type == "cpu"


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #


def test_fit_actually_descends_the_loss() -> None:
    """The one thing a hand-written loop can get silently wrong.

    A loop that forgets `backward()` or `step()` still runs, still logs, and still reports
    metrics — it just never learns. So this asserts the loss MOVES rather than that fit returned.
    """
    clf = _clf(train_set=Stream(source=_records(24)), max_epochs=4, num_classes=NUM_CLASSES)
    clf.run()

    losses = clf._history["train/loss"]
    assert len(losses) == 4
    assert losses[-1] < losses[0], f"the loss did not descend: {losses}"


def test_the_weights_change() -> None:
    """The complement of the loss check: gradients reach the parameters, not just the graph."""
    model = _Tiny()
    before = model.fc.weight.detach().clone()
    clf = _clf(model=model, train_set=Stream(source=_records(16)), max_epochs=1, num_classes=NUM_CLASSES)
    clf.run()

    assert not torch.allclose(before, model.fc.weight)


def test_a_wired_val_set_is_scored_every_epoch() -> None:
    clf = _clf(train_set=Stream(source=_records(16)), max_epochs=2, num_classes=NUM_CLASSES)
    clf.val_set = Stream(source=_records(8))
    clf.run()

    assert len(clf._history["val/loss"]) == 2
    assert "val/accuracy_top1" in clf.computed_metrics


def test_metrics_are_reset_between_epochs() -> None:
    """Forgetting the reset is invisible in the numbers' SHAPE — they just stop being per-epoch,
    silently averaging every batch since the run began."""
    clf = _clf(train_set=Stream(source=_records(16)), max_epochs=3, num_classes=NUM_CLASSES)
    clf.initialize()
    resets: List[int] = []
    collection = clf.train_metrics
    assert collection is not None
    original = collection.reset

    def _spy() -> None:
        resets.append(1)
        original()

    setattr(collection, "reset", _spy)
    clf.fit()

    assert len(resets) == 3, f"one reset per epoch, got {len(resets)}"


def test_a_confusion_matrix_is_rendered_rather_than_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    """`_epoch_metrics` returns SCALARS — a `Tracker.log_metrics` channel takes nothing else.

    A non-scalar metric would therefore vanish without a trace, which is worse than not
    supporting it. It is rendered into the log instead; the scalars beside it are unaffected.
    Patched on `sonair.torch_classifier`, the module that BINDS the name.
    """
    from confluid import LazyClass
    from torchmetrics import Metric
    from torchmetrics.classification import MulticlassConfusionMatrix

    from sonair import torch_classifier as mod

    rendered: List[Any] = []
    monkeypatch.setattr(mod, "render_matrix_as_table", lambda name, matrix, names: rendered.append(name))

    clf = _clf(train_set=Stream(source=_records(16)), num_classes=NUM_CLASSES)
    clf.metrics = {"confusion": LazyClass[Metric](MulticlassConfusionMatrix)}
    clf.initialize()
    collection = clf.train_metrics
    assert collection is not None
    collection.update(torch.randn(8, NUM_CLASSES), torch.randint(0, NUM_CLASSES, (8,)))

    assert clf._epoch_metrics(collection) == {}  # nothing scalar to report
    assert rendered == ["train/confusion"]


def test_a_rendered_matrix_reaches_every_wired_tracker(monkeypatch: pytest.MonkeyPatch) -> None:
    """The log is not the only destination — `log_text` is the channel that exists for this.

    Without it a tracked run records the scalars and nothing else, which reads as a complete
    record rather than a missing one.
    """
    from confluid import LazyClass
    from torchmetrics import Metric
    from torchmetrics.classification import MulticlassConfusionMatrix

    class _Recorder:
        def __init__(self) -> None:
            self.texts: List[Any] = []

        def log_text(self, name: str, text: str, step: Any = None) -> None:
            self.texts.append((name, text, step))

    tracker = _Recorder()
    clf = _clf(train_set=Stream(source=_records(16)), num_classes=NUM_CLASSES)
    clf.metrics = {"confusion": LazyClass[Metric](MulticlassConfusionMatrix)}
    clf.initialize()
    collection = clf.val_metrics
    assert collection is not None
    collection.update(torch.randn(8, NUM_CLASSES), torch.randint(0, NUM_CLASSES, (8,)))

    clf._epoch_metrics(collection, [tracker], step=2)  # type: ignore[list-item]

    assert len(tracker.texts) == 1
    name, table, step = tracker.texts[0]
    assert (name, step) == ("val/confusion", 2)
    assert table.startswith("val/confusion:") and "pred:0" in table


def test_a_split_slot_adds_to_the_shared_set_for_that_split_only() -> None:
    """Same rule as the Lightning backend — a config moving between them must not change shape."""
    from confluid import LazyClass
    from torchmetrics import Metric
    from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix

    clf = _clf(
        train_set=Stream(source=_records(16)),
        num_classes=NUM_CLASSES,
        metrics={"accuracy_top1": LazyClass[Metric](MulticlassAccuracy, top_k=1)},
        val_metrics={"confusion": LazyClass[Metric](MulticlassConfusionMatrix)},
    )
    clf.initialize()

    assert sorted(str(k) for k in clf.val_metrics.keys()) == ["val/accuracy_top1", "val/confusion"]
    assert [str(k) for k in clf.train_metrics.keys()] == ["train/accuracy_top1"]


def test_the_eval_pass_puts_the_model_in_eval_mode() -> None:
    """`model.eval()` and `torch.no_grad()` are the loop's job here — dropout/batch-norm depend
    on the first, and the second is what keeps evaluation from building a graph."""
    seen: List[bool] = []

    class _Probe(_Tiny):
        def forward(self, x: Any) -> Any:
            seen.append(self.training)
            out = super().forward(x)
            assert out.requires_grad == self.training, "eval must not build an autograd graph"
            return out

    clf = _clf(model=_Probe(), train_set=Stream(source=_records(8)), max_epochs=1, num_classes=NUM_CLASSES)
    clf.val_set = Stream(source=_records(8))
    clf.run()

    assert True in seen and False in seen, "both a training and an eval pass must have run"


def test_grad_clipping_is_applied_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[float] = []
    original = torch.nn.utils.clip_grad_norm_

    def _spy(params: Any, max_norm: float, *a: Any, **kw: Any) -> Any:
        calls.append(max_norm)
        return original(params, max_norm, *a, **kw)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", _spy)
    clf = _clf(train_set=Stream(source=_records(8)), max_epochs=1, num_classes=NUM_CLASSES, grad_clip=1.0)
    clf.run()

    assert calls and set(calls) == {1.0}


def test_the_deferred_loader_slots_drive_the_run() -> None:
    """`batch_size` reaches the DataLoader through the `LazyClass`, so a config can override any
    loader knob this class never enumerates."""
    clf = _clf(train_set=Stream(source=_records(12)), batch_size=3, num_classes=NUM_CLASSES)
    clf.initialize()
    loader = clf._loaders()["train"]

    assert loader is not None and loader.batch_size == 3
    assert len(list(loader)) == 4


# --------------------------------------------------------------------------- #
# Labels, defaults and balancing — parity with the Lightning backend
# --------------------------------------------------------------------------- #


def test_string_targets_are_encoded_and_num_classes_derived() -> None:
    clf = _clf(train_set=Stream(source=_records(9, names=True)), max_epochs=1)
    clf.run()

    assert clf.num_classes == NUM_CLASSES
    assert clf.class_names == ["a", "b", "c"]


def test_the_loss_and_metric_defaults_match_the_lightning_backend() -> None:
    """A config moved between the two backends must not silently change its loss."""
    pytest.importorskip("pytorch_lightning", reason="needs sonair[lightning] for the comparison")
    from sonair.lightning_classifier import LightningClassifier

    for multilabel in (False, True):
        mine, theirs = _clf(), LightningClassifier(model=_Tiny())
        mine.multilabel = theirs.multilabel = multilabel
        mine._resolve_task_defaults()
        theirs._resolve_task_defaults()

        assert type(mine.loss_fn) is type(theirs.loss_fn)
        assert sorted(mine.metrics) == sorted(theirs.metrics)


def test_an_explicitly_wired_loss_is_never_replaced() -> None:
    """That is what the `None` sentinel buys: "not chosen" is distinguishable from "chose the
    single-label default"."""
    clf = _clf()
    clf.loss_fn = nn.BCEWithLogitsLoss()
    clf._resolve_task_defaults()

    assert isinstance(clf.loss_fn, nn.BCEWithLogitsLoss)


def test_class_weights_are_injected_into_a_weight_capable_loss() -> None:
    """The STATISTIC is recordstream's; putting it into the loss is this backend's job."""
    skewed = _records(4) + _records(8)[:8]  # class 0 over-represented
    clf = _clf(train_set=Stream(source=skewed), max_epochs=1, balance_classes=True)
    clf.initialize()

    assert clf.class_weights is not None
    assert len(clf.class_weights) == NUM_CLASSES
    assert getattr(clf.loss_fn, "weight", None) is not None


def test_balancing_is_skipped_for_a_loss_that_cannot_take_weights() -> None:
    """`accepts_class_weights` also GATES the target walk, so it must answer before the loss
    is built — and answering wrongly would mean a walk whose result nothing can use."""

    class _NoWeight(nn.Module):
        def forward(self, logits: Any, target: Any) -> Any:  # pragma: no cover - never called
            return logits.sum()

    clf = _clf(train_set=Stream(source=_records(8)), balance_classes=True)
    clf.loss_fn = _NoWeight()

    assert clf._accepts_class_weights() is False
    clf.initialize()
    assert clf.class_weights is None


# --------------------------------------------------------------------------- #
# Evaluate, predict and the artifacts
# --------------------------------------------------------------------------- #


def test_test_scores_a_wired_model_and_writes_the_sidecar(tmp_path: Path) -> None:
    sidecar = tmp_path / "metrics.json"
    clf = _clf(test_set=Stream(source=_records(12)), task="test")
    clf.metrics_sidecar_path = str(sidecar)
    clf.run()

    assert "test/loss" in clf.computed_metrics and "test/accuracy_top1" in clf.computed_metrics
    assert sidecar.exists()


def test_num_classes_comes_from_the_model_on_the_eval_path() -> None:
    """The trained head is authoritative — an eval split may never show every class, and a
    dataset-derived count would mis-size the metrics against the logits."""
    clf = _clf(model=_Tiny(num_classes=7), test_set=Stream(source=_records(6)), task="test")
    clf._initialize_eval()

    assert clf.num_classes == 7


def test_predict_writes_ONE_row_per_record_with_its_metadata() -> None:
    """The sink contract is per-record while the model emits `[N, C]` — handing the batch over
    whole would write one entry for a whole batch."""
    clf = _clf(test_set=Stream(source=_records(10)), task="predict", batch_size=4)
    sink = _CaptureSink()
    clf.predictions_sink = sink
    clf.run()

    assert len(sink.rows) == 10 and sink.closed
    first, meta = sink.rows[0]
    assert sorted(first) == ["class_idx", "logits", "probs"]
    assert first["probs"].shape == (NUM_CLASSES,)
    assert pytest.approx(float(first["probs"].sum()), abs=1e-5) == 1.0
    assert meta == {"row_id": 0}


def test_a_checkpoint_dir_gets_the_weights_and_the_class_names(tmp_path: Path) -> None:
    clf = _clf(
        train_set=Stream(source=_records(8, names=True)),
        max_epochs=1,
        checkpoint_dir=str(tmp_path / "run"),
    )
    clf.run()

    assert (tmp_path / "run" / "model.pt").exists()
    assert (tmp_path / "run" / "class_names.json").exists()
    # A plain state_dict, not a framework container — there is no framework.
    assert set(torch.load(tmp_path / "run" / "model.pt")) == {"fc.weight", "fc.bias"}


def test_the_outputs_raise_before_the_run() -> None:
    from confluid import LazyClass

    clf = TorchClassifier(model=LazyClass(_Tiny))
    with pytest.raises(RuntimeError, match="only available after run"):
        _ = clf.trained_model


def test_a_non_module_in_the_model_slot_fails_during_setup() -> None:
    """During SETUP, not minutes into a fit — the failure a wrong `model:` should produce."""
    clf = _clf(model="not-a-module", train_set=Stream(source=_records(4)), num_classes=NUM_CLASSES)

    with pytest.raises(TypeError, match="must be a torch.nn.Module"):
        clf.initialize()


# --------------------------------------------------------------------------- #
# The config surface
# --------------------------------------------------------------------------- #


def test_the_constructor_mirrors_the_lightning_backends_knobs() -> None:
    """One config drives either backend, so the shared knob NAMES have to line up."""
    pytest.importorskip("pytorch_lightning", reason="needs sonair[lightning] for the comparison")
    from sonair.lightning_classifier import LightningClassifier

    theirs = set(inspect.signature(LightningClassifier.__init__).parameters)
    mine = set(inspect.signature(TorchClassifier.__init__).parameters)

    # Nothing Lightning-only remains: the `logger` slot of `pl.loggers.Logger`s is GONE, and
    # both backends take the same `trackers` — which is the point of removing it.
    assert theirs - mine == set()
    assert mine - theirs == {"device", "checkpoint_dir", "grad_clip"}
    assert "trackers" in theirs and "trackers" in mine
    assert "logger" not in theirs and "logger" not in mine


def test_the_run_dispatch_covers_every_task() -> None:
    from recordstream.runnable import runnable_entrypoints

    assert {"fit", "evaluate", "test", "predict"} <= set(runnable_entrypoints(TorchClassifier))


# --------------------------------------------------------------------------- #
# tracking: one derivation, shared with the Lightning backend
# --------------------------------------------------------------------------- #


def test_provenance_derives_the_hyperparameters_from_the_runnable(monkeypatch: pytest.MonkeyPatch) -> None:
    """This backend must not hand-write a params dict.

    It used to spell six keys by hand while the Lightning backend walked the object graph,
    so the two gave different answers to "what were the hyperparameters?". Both now hand the
    runnable to marainer, which owns the one derivation.
    """
    from marainer.tracking import provenance

    seen: Dict[str, Any] = {}

    def spy(trackers: Any, params: Any = None, **kwargs: Any) -> Any:
        seen["params"] = params
        seen["runnable"] = kwargs.get("runnable")
        return None

    monkeypatch.setattr(provenance, "log_run_context", spy)
    trainer = _clf(max_epochs=7)
    trainer._log_run_context([CsvTracker()])

    assert seen["runnable"] is trainer  # the object, not a dict transcribed from it
    # `device` is the one value that is NOT config — it is resolved at run time.
    assert set(seen["params"]) == {"device"}


def test_provenance_is_skipped_without_trackers() -> None:
    assert _clf()._log_run_context([]) is None


def test_fit_finishes_the_manifest(tmp_path: Path) -> None:
    """A finished run must not be left at `running` for the atexit hook to call interrupted."""
    import yaml

    clf = _clf(train_set=_records(8), max_epochs=1)
    # `trackers` is a post-construction slot, set the way confluid would.
    clf.trackers = [CsvTracker(experiment_name="e", run_name="r", save_dir=str(tmp_path))]
    clf.run()
    manifest = yaml.safe_load((tmp_path / "e" / "r" / "manifest.yaml").read_text())
    assert manifest["status"] == "success"
    assert manifest["params"]  # derived from the runnable, not hand-written
