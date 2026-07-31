# mypy: disable-error-code="attr-defined"
"""``TensorFlowClassifier`` — the native tf.data + GradientTape backend.

Two groups of tests, split by what they need:

* **Source-only** — the import ordering. These read the module as TEXT and run EVERYWHERE,
  including on an install with no TensorFlow at all. That is deliberate and it is the point:
  the ordering they guard decides which engine keras loads, and getting it wrong produces a
  symptom (``tf.GradientTape`` failing on a torch model) very far from its cause. A guard that
  only ran where TensorFlow is installed would not be watching the machines that can break it.
* **Behavioural** — everything that imports the module, skipped unless keras is on the
  TensorFlow engine in THIS process.

That last condition is not a hedge, it is Keras's own rule: **Keras 3 binds exactly one engine
per process, at first import**. A full-suite run imports ``tests/test_keras_classifier.py``
first (alphabetically), which loads keras on torch — so the TensorFlow loop genuinely cannot run
in the same process, and a marker that pretended otherwise would report coverage that never
executed. Run them as their own invocation::

    KERAS_BACKEND=tensorflow pytest tests/test_tensorflow_classifier.py

Running this file alone needs no environment variable: it imports ``sonair._tensorflow`` before
anything else can claim the engine.

TensorFlow is also not part of the workspace's *selected* frameworks by default (it is a large
download, and nothing else needs it), so this group skips entirely on a default checkout and
runs on one that opted in with ``aisland framework set … tensorflow``.
"""

import importlib.util
import re
import sys
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parent.parent / "sonair" / "tensorflow_classifier.py"

# Claim the engine before any other import can — this is what lets `pytest
# tests/test_tensorflow_classifier.py` work with no environment variable set. Guarded on keras
# being unimported, because if something already bound it the import would be a no-op that
# merely costs a TensorFlow load.
if importlib.util.find_spec("tensorflow") is not None and "keras" not in sys.modules:
    import sonair._tensorflow  # noqa: F401


def _keras_is_on_tensorflow() -> bool:
    """Whether keras is bound to the TensorFlow engine in this process (never importing it)."""
    keras = sys.modules.get("keras")
    return keras is not None and keras.backend.backend() == "tensorflow"


#: Applied per-test rather than to the module, so the source-only group above still runs where
#: TensorFlow is missing — see the module docstring for why that matters.
requires_tensorflow = pytest.mark.skipif(
    not _keras_is_on_tensorflow(),
    reason="needs keras on the TensorFlow engine: KERAS_BACKEND=tensorflow pytest tests/test_tensorflow_classifier.py",
)


# ---------------------------------------------------------------------------
# Source-only: import ordering (runs without TensorFlow too — see the module docstring)
# ---------------------------------------------------------------------------


def _sonair_imports(source: str) -> List[str]:
    """Every ``from sonair.X import …`` module named in ``source``, in source order."""
    return re.findall(r"^from (sonair\.[\w.]+) import", source, flags=re.MULTILINE)


def test_keras_is_imported_through_the_tensorflow_boundary() -> None:
    """``sonair._keras`` must not appear here — it would load keras on the WRONG engine.

    Keras reads ``KERAS_BACKEND`` once, at import time, and ``recordstream.keras`` (which
    ``sonair._keras`` re-exports) defaults it to the first backend INSTALLED — torch, on any
    machine that also carries ``sonair[lightning]``. isort sorts ``sonair._keras`` ABOVE
    ``sonair._tensorflow``, so importing the handle from its usual home would beat the
    ``tensorflow`` default into place and leave ``tf.GradientTape`` driving a torch model.
    """
    imports = _sonair_imports(SOURCE.read_text())
    assert "sonair._keras" not in imports, "import keras from sonair._tensorflow, which pins the engine first"
    assert "sonair._tensorflow" in imports


def test_the_tensorflow_boundary_is_imported_before_anything_that_loads_keras() -> None:
    """``sonair._tensorflow`` must be the FIRST first-party import in the module."""
    imports = _sonair_imports(SOURCE.read_text())
    assert imports[0] == "sonair._tensorflow", f"sonair._tensorflow must come first, got {imports}"


def test_the_boundary_module_sets_the_backend_before_importing_anything() -> None:
    """The ``setdefault`` in ``sonair/_tensorflow.py`` must precede its own imports."""
    source = (SOURCE.parent / "_tensorflow.py").read_text()
    setdefault = source.index('os.environ.setdefault("KERAS_BACKEND"')
    assert setdefault < source.index("import tensorflow")
    assert setdefault < source.index("from recordstream.keras import")


# ---------------------------------------------------------------------------
# Static: what the subclass actually adds
# ---------------------------------------------------------------------------


@requires_tensorflow
def test_it_subclasses_the_keras_backend() -> None:
    """In TF >= 2.16 ``tf.keras`` IS Keras 3 — the label handling and batch shape are the same
    code, not a copy of it. If this ever stops being true the reuse claim in the docs is wrong."""
    from sonair.keras_classifier import KerasClassifier
    from sonair.tensorflow_classifier import TensorFlowClassifier

    assert issubclass(TensorFlowClassifier, KerasClassifier)


@requires_tensorflow
def test_it_overrides_only_the_tensorflow_specific_behaviour() -> None:
    """Pins the SIZE of the divergence — the point of the backend, and easy to erode silently.

    Anything added here should be about tf.data, the traced loop, or the SavedModel. A method
    that is really about classification belongs on the parent, where every backend gets it.
    """
    from sonair.tensorflow_classifier import TensorFlowClassifier

    # Callables only: the class body's methods, minus the string attributes `@configurable`
    # stamps on (`__confluid_task__` and friends) and the ones Python adds (`__module__`, …).
    overrides = {name for name, value in vars(TensorFlowClassifier).items() if callable(value)}
    assert overrides == {
        "__init__",
        "initialize",  # + the engine check
        "_require_tensorflow_backend",
        "_resolve_task_defaults",  # objects, not aliases — no compile() to interpret them
        "_compile",  # builds the loop's parts instead of Model.compile
        "_dataset",  # tf.data
        "_train_step_fn",
        "_eval_step_fn",
        "_predict_step_fn",
        "_reset_metrics",
        "_metric_results",
        "_run_epoch",
        "fit",
        "test",
        "predict",
        "_export",  # SavedModel
    }


@requires_tensorflow
def test_the_constructor_restates_every_knob_of_its_parent() -> None:
    """``**kwargs`` would type-check but erase the config surface that `to_pydantic` reads."""
    import inspect

    from sonair.keras_classifier import KerasClassifier
    from sonair.tensorflow_classifier import TensorFlowClassifier

    parent = set(inspect.signature(KerasClassifier.__init__).parameters)
    child = set(inspect.signature(TensorFlowClassifier.__init__).parameters)
    assert parent <= child, f"dropped from the config surface: {sorted(parent - child)}"
    assert child - parent == {"export_dir", "jit_compile", "prefetch"}


@requires_tensorflow
def test_discovery_tags() -> None:
    """``framework="keras"`` is deliberate: the axis names the API a class is WIRED TO (a
    ``keras.losses.Loss``, a ``KerasModel``), not the tensor runtime underneath."""
    from sonair.tensorflow_classifier import TensorFlowClassifier

    assert TensorFlowClassifier.__confluid_task__ == "classification"
    assert TensorFlowClassifier.__confluid_role__ == "trainer"
    assert TensorFlowClassifier.__confluid_framework__ == "keras"


@requires_tensorflow
def test_zero_arg_construction_does_no_work() -> None:
    """The workspace lazy-init mandate: the constructor stores, it does not build."""
    from sonair.tensorflow_classifier import TensorFlowClassifier

    clf = TensorFlowClassifier(model=None)
    assert clf.task == "fit"
    assert clf.num_classes is None
    assert clf._optimizer is None and clf._loss_fn is None


# ---------------------------------------------------------------------------
# Behavioural
# ---------------------------------------------------------------------------


@pytest.fixture()
def keras_module() -> Any:
    from sonair._tensorflow import keras

    return keras


class _Shim:
    """The ``.model`` + ``.num_classes`` shape a model wrapper exposes (what ``KerasModel`` is)."""

    def __init__(self, model: Any, num_classes: int) -> None:
        self.model = model
        self.num_classes = num_classes


class _CaptureSink:
    def __init__(self) -> None:
        self.rows: List[Tuple[Any, Any]] = []
        self.closed = False

    def write(self, prediction: Any, metadata: Any) -> None:
        self.rows.append((prediction, metadata))

    def close(self) -> None:
        self.closed = True


def _records(n: int, num_classes: int = 3) -> List[dict]:
    """Channels-LAST records — the shape a Keras/TensorFlow preprocessing chain produces."""
    from recordstream.items import Image

    rng = np.random.default_rng(0)
    return [
        {"image": Image(rng.random((8, 8, 3), dtype=np.float32)), "class": i % num_classes, "row_id": i}
        for i in range(n)
    ]


def _dense_model(keras: Any, num_classes: int = 3) -> Any:
    return keras.Sequential([keras.layers.Input((8, 8, 3)), keras.layers.Flatten(), keras.layers.Dense(num_classes)])


@requires_tensorflow
def test_the_backend_check_names_the_engine_it_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wrong engine -> a message naming the fix, not a failure inside the tape."""
    import sonair.tensorflow_classifier as module

    monkeypatch.setattr(module, "keras_backend", lambda: "torch")
    with pytest.raises(RuntimeError, match="KERAS_BACKEND"):
        module.TensorFlowClassifier._require_tensorflow_backend()


@requires_tensorflow
def test_fit_runs_the_graph_loop_and_records_history(keras_module: Any) -> None:
    from recordstream import Stream

    from sonair.tensorflow_classifier import TensorFlowClassifier

    clf = TensorFlowClassifier(
        model=_Shim(_dense_model(keras_module), 3),
        train_set=Stream(source=_records(12)),
        num_classes=3,
        max_epochs=2,
        batch_size=4,
    )
    clf.run()

    assert set(clf._history) == {"loss", "sparse_categorical_accuracy"}
    assert len(clf._history["loss"]) == 2
    assert clf.computed_metrics["loss"] == clf._history["loss"][-1]


@requires_tensorflow
def test_a_wired_val_set_is_scored_every_epoch(keras_module: Any) -> None:
    from recordstream import Stream

    from sonair.tensorflow_classifier import TensorFlowClassifier

    clf = TensorFlowClassifier(
        model=_Shim(_dense_model(keras_module), 3),
        train_set=Stream(source=_records(12)),
        num_classes=3,
        max_epochs=1,
        batch_size=4,
    )
    clf.val_set = Stream(source=_records(8))
    clf.run()

    assert "val_loss" in clf.computed_metrics
    assert "val_sparse_categorical_accuracy" in clf.computed_metrics


@requires_tensorflow
def test_keras_callbacks_are_driven_by_hand(keras_module: Any, tmp_path: Path) -> None:
    """A hand-written loop has no ``fit()`` to run callbacks, so it drives a ``CallbackList``.

    ``CSVLogger`` is the check because it exercises the whole protocol — ``on_train_begin``
    opens the file, ``on_epoch_end`` writes a row, ``on_train_end`` closes it. Its parent
    directory is created for it, the same defence the Keras backend added after a real run died
    minutes into training.
    """
    from recordstream import Stream

    from sonair.tensorflow_classifier import TensorFlowClassifier

    target = tmp_path / "not_yet" / "history.csv"
    clf = TensorFlowClassifier(
        model=_Shim(_dense_model(keras_module), 3),
        train_set=Stream(source=_records(8)),
        num_classes=3,
        max_epochs=2,
        batch_size=4,
        logger=[keras_module.callbacks.CSVLogger(filename=str(target))],
    )
    clf.run()

    rows = target.read_text().strip().splitlines()
    assert rows[0].startswith("epoch,")
    assert len(rows) == 3, "a header plus one row per epoch"


@requires_tensorflow
def test_shuffling_reshuffles_between_epochs(keras_module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """``PyDataset.on_epoch_end`` is Keras's job under ``fit()`` and the loop's job here.

    Forgetting it is invisible — training still converges, just on one fixed row order — so it
    gets an explicit test rather than trusting the comment beside it.
    """
    from recordstream import Stream
    from recordstream.keras import RecordSequence

    from sonair.tensorflow_classifier import TensorFlowClassifier

    reshuffles: List[bool] = []
    original = RecordSequence.on_epoch_end

    def _spy(self: Any) -> None:
        reshuffles.append(self.shuffle)
        original(self)

    monkeypatch.setattr(RecordSequence, "on_epoch_end", _spy)

    clf = TensorFlowClassifier(
        model=_Shim(_dense_model(keras_module), 3),
        train_set=Stream(source=_records(8)),
        num_classes=3,
        max_epochs=3,
        batch_size=4,
    )
    clf.run()

    assert reshuffles == [True, True, True], "the training sequence must be reshuffled after every epoch"


@requires_tensorflow
def test_the_default_loss_and_metrics_are_objects_not_aliases(keras_module: Any) -> None:
    """No ``compile()`` means no contextual alias resolution — see ``_resolve_task_defaults``."""
    from sonair.tensorflow_classifier import TensorFlowClassifier

    clf = TensorFlowClassifier(model=None)
    clf._resolve_task_defaults()

    assert isinstance(clf.loss, keras_module.losses.SparseCategoricalCrossentropy)
    assert clf.loss.from_logits is True, "the head is a bare Dense — logits, not probabilities"
    assert all(isinstance(m, keras_module.metrics.Metric) for m in clf.metrics or [])


@requires_tensorflow
def test_multilabel_defaults_switch_both_slots(keras_module: Any) -> None:
    from sonair.tensorflow_classifier import TensorFlowClassifier

    clf = TensorFlowClassifier(model=None)
    clf.multilabel = True
    clf._resolve_task_defaults()

    assert isinstance(clf.loss, keras_module.losses.BinaryCrossentropy)
    assert isinstance((clf.metrics or [None])[0], keras_module.metrics.BinaryAccuracy)


@requires_tensorflow
def test_an_explicitly_wired_loss_is_never_replaced(keras_module: Any) -> None:
    from sonair.tensorflow_classifier import TensorFlowClassifier

    chosen = keras_module.losses.CategoricalCrossentropy(from_logits=True)
    clf = TensorFlowClassifier(model=None, loss=chosen)
    clf._resolve_task_defaults()
    clf._resolve_task_defaults()  # idempotent

    assert clf.loss is chosen


@requires_tensorflow
def test_the_accuracy_alias_resolves_the_way_compile_would(keras_module: Any) -> None:
    """``keras.metrics.get("accuracy")`` is elementwise equality — 0.0 forever on a logits head."""
    from sonair.tensorflow_classifier import _as_metric

    assert isinstance(_as_metric("accuracy", multilabel=False), keras_module.metrics.SparseCategoricalAccuracy)
    assert isinstance(_as_metric("accuracy", multilabel=True), keras_module.metrics.BinaryAccuracy)


@pytest.mark.parametrize(
    "value",
    ["sparse_categorical_accuracy", "SparseCategoricalAccuracy"],
    ids=["function-alias", "class-name-alias"],
)
@requires_tensorflow
def test_other_metric_aliases_still_resolve_to_metric_objects(keras_module: Any, value: str) -> None:
    from sonair.tensorflow_classifier import _as_metric

    assert isinstance(_as_metric(value, multilabel=False), keras_module.metrics.Metric)


@requires_tensorflow
def test_a_metric_class_is_instantiated(keras_module: Any) -> None:
    from sonair.tensorflow_classifier import _as_metric

    resolved = _as_metric(keras_module.metrics.SparseCategoricalAccuracy, multilabel=False)
    assert isinstance(resolved, keras_module.metrics.SparseCategoricalAccuracy)


@requires_tensorflow
@pytest.mark.parametrize("value", ["adam", "Adam"], ids=["lowercase", "class-name"])
def test_optimizer_aliases_resolve_to_instances(keras_module: Any, value: str) -> None:
    from sonair.tensorflow_classifier import _as_optimizer

    assert isinstance(_as_optimizer(value), keras_module.optimizers.Optimizer)


@requires_tensorflow
def test_the_output_signature_follows_the_real_batch() -> None:
    """Derived from batch zero so the batch SHAPE stays defined only in ``_to_arrays``."""
    from sonair.tensorflow_classifier import _output_signature

    specs = _output_signature((np.zeros((4, 8, 8, 3), dtype="float32"), np.zeros((4,), dtype="int64")))
    assert [tuple(s.shape) for s in specs] == [(None, 8, 8, 3), (None,)]
    assert [s.dtype.name for s in specs] == ["float32", "int64"]


@requires_tensorflow
def test_evaluate_scores_the_test_set(keras_module: Any) -> None:
    from recordstream import Stream

    from sonair.tensorflow_classifier import TensorFlowClassifier

    clf = TensorFlowClassifier(
        model=_Shim(_dense_model(keras_module), 3),
        test_set=Stream(source=_records(8)),
        task="test",
        batch_size=4,
    )
    clf.run()

    assert set(clf.computed_metrics) == {"loss", "sparse_categorical_accuracy"}


@requires_tensorflow
def test_predict_writes_one_entry_per_record_with_its_metadata(keras_module: Any) -> None:
    """The sink contract is per-record while the model emits ``[N, C]`` — and the metadata that
    travelled beside the image has to survive the trip."""
    from recordstream import Stream

    from sonair.tensorflow_classifier import TensorFlowClassifier

    sink = _CaptureSink()
    clf = TensorFlowClassifier(
        model=_Shim(_dense_model(keras_module), 3),
        test_set=Stream(source=_records(7)),
        task="predict",
        batch_size=3,
    )
    clf.predictions_sink = sink  # type: ignore[assignment]
    clf.run()

    assert len(sink.rows) == 7, "a short final batch must not be dropped or padded"
    prediction, metadata = sink.rows[0]
    assert set(prediction) == {"logits", "probs", "class_idx"}
    assert metadata == {"row_id": 0}
    assert pytest.approx(float(sum(prediction["probs"])), abs=1e-5) == 1.0
    assert sink.closed


@requires_tensorflow
def test_export_writes_a_savedmodel(keras_module: Any, tmp_path: Path) -> None:
    """The TensorFlow-only artifact, and the concrete reason to pick this backend."""
    from recordstream import Stream

    from sonair.tensorflow_classifier import TensorFlowClassifier

    target = tmp_path / "exported" / "model"
    clf = TensorFlowClassifier(
        model=_Shim(_dense_model(keras_module), 3),
        train_set=Stream(source=_records(8)),
        num_classes=3,
        max_epochs=1,
        batch_size=4,
        export_dir=str(target),
    )
    clf.run()

    assert (target / "saved_model.pb").exists()


@requires_tensorflow
def test_no_export_dir_writes_nothing(keras_module: Any, tmp_path: Path) -> None:
    from recordstream import Stream

    from sonair.tensorflow_classifier import TensorFlowClassifier

    clf = TensorFlowClassifier(
        model=_Shim(_dense_model(keras_module), 3),
        train_set=Stream(source=_records(8)),
        num_classes=3,
        max_epochs=1,
        batch_size=4,
    )
    clf.run()

    assert not list(tmp_path.iterdir())


@requires_tensorflow
def test_fit_without_a_train_set_is_a_clear_error(keras_module: Any) -> None:
    from sonair.tensorflow_classifier import TensorFlowClassifier

    clf = TensorFlowClassifier(model=_Shim(_dense_model(keras_module), 3), num_classes=3)
    clf._compile()
    with pytest.raises(ValueError, match="train_set"):
        clf.fit()


@requires_tensorflow
@pytest.mark.parametrize("method", ["test", "predict"])
def test_eval_verbs_without_a_test_set_are_clear_errors(keras_module: Any, method: str) -> None:
    from sonair.tensorflow_classifier import TensorFlowClassifier

    clf = TensorFlowClassifier(model=_Shim(_dense_model(keras_module), 3), num_classes=3)
    clf._compile()
    with pytest.raises(ValueError, match="test_set"):
        getattr(clf, method)()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
