"""The Keras loss/metric registrations — the ``framework="keras"`` half of the picker.

These pin three things a broken registration would silently cost:

1. every class in ``_CLASSIFICATION_LOSSES`` / ``_CLASSIFICATION_METRICS`` is discoverable
   under ``task="classification"`` + its role + ``framework="keras"`` (a dropped tag empties
   a picker rather than failing anything);
2. ``__all__`` names every one of them, because ``scan_module`` filters by ``__module__``
   (``keras.src.*`` here) and the ``__all__``-gated pass is what surfaces them to StreamStudio;
3. the seven loss/metric NAMESAKES resolve to the right class per ``role`` — Keras publishes
   thirteen names under both ``keras.losses`` and ``keras.metrics``, and until confluid
   admitted a shared name (0.3.0) registering the metric silently overwrote the loss, so
   those seven could not be registered at all.

Plus the numeric facts the module docstring's wiring guidance rests on: which metrics are
safe on the classifier's raw-logits head, and what to set for the ones that are not.
"""

import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import confluid
import numpy as np
import pytest
from confluid import AmbiguousClassError, Fluid, get_registry

import sonair.keras_classification as kc
from sonair._keras import keras
from sonair.keras_classifier import KerasClassifier

# 64x4 logits and their sigmoid — a multi-label decision surface, which is where the
# threshold-based metrics actually apply.
_RNG = np.random.default_rng(0)
LOGITS = (_RNG.normal(size=(64, 4)) * 2).astype("float32")
PROBS = 1.0 / (1.0 + np.exp(-LOGITS))
TARGETS = (_RNG.random((64, 4)) < 0.3).astype("float32")
#: The single-label spelling of the same batch — what the ``Sparse*`` metrics consume.
IDS = TARGETS.argmax(axis=-1).astype("int64")


def _value(metric: object, y_pred: np.ndarray, y_true: np.ndarray = TARGETS) -> float:
    metric.update_state(y_true, y_pred)  # type: ignore[attr-defined]
    return round(float(metric.result()), 6)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------


def test_every_keras_loss_is_registered_under_its_framework() -> None:
    """Asserted through the FILTERED lookup, which is the picker's actual query.

    A name shared with a metric publishes under its dotted key, so a bare membership test
    would fail for exactly the seven classes this change made registrable.
    """
    registry = get_registry()
    for cls in kc._CLASSIFICATION_LOSSES:
        found = registry.get_class(cls.__name__, task="classification", role="loss", framework="keras")
        assert found is cls, f"{cls.__name__} not registered as a keras classification loss"
    assert len(registry.list_classes(task="classification", role="loss", framework="keras")) == len(
        kc._CLASSIFICATION_LOSSES
    )


def test_every_keras_metric_is_registered_under_its_framework() -> None:
    registry = get_registry()
    for cls in kc._CLASSIFICATION_METRICS:
        found = registry.get_class(cls.__name__, task="classification", role="metric", framework="keras")
        assert found is cls, f"{cls.__name__} not registered as a keras classification metric"
    # category is the derived `classification_metric`, which navigaitor's metric-slot picker reads.
    assert "SparseCategoricalAccuracy" in registry.list_classes(category="classification_metric")


def test_all_exports_registered_losses_and_metrics() -> None:
    """``__all__`` must name every registered class or StreamStudio's pass never sees it."""
    for cls in (*kc._CLASSIFICATION_LOSSES, *kc._CLASSIFICATION_METRICS):
        assert cls.__name__ in kc.__all__, f"{cls.__name__} registered but missing from __all__"
    assert len(kc.__all__) == len(set(kc.__all__))


def test_the_loss_metric_namesakes_are_both_registered_and_resolve_by_role() -> None:
    """The case this whole registration set was blocked on.

    Keras publishes these names under BOTH ``keras.losses`` and ``keras.metrics``. The
    registry keys by name, so registering the metric used to overwrite the loss — silently,
    since both are ``task="classification"`` configurables. It now admits both because
    ``role`` tells them apart.
    """
    registry = get_registry()
    shared = {n for n in dir(keras.losses) if n[0].isupper()} & {n for n in dir(keras.metrics) if n[0].isupper()}
    namesakes = {c.__name__ for c in kc._CLASSIFICATION_LOSSES} & {c.__name__ for c in kc._CLASSIFICATION_METRICS}
    assert namesakes <= shared and len(namesakes) == 7, namesakes

    for name in namesakes:
        loss = registry.get_class(name, role="loss", framework="keras")
        metric = registry.get_class(name, role="metric", framework="keras")
        assert loss is getattr(keras.losses, name)
        assert metric is getattr(keras.metrics, name)
        assert loss is not metric
        # ...and a BARE reference is an error naming both, not a coin flip.
        with pytest.raises(AmbiguousClassError):
            registry.get_class(name)


def test_registering_the_UPSTREAM_keras_set_emits_no_clobber_warning() -> None:
    """Every duplicate among the registered upstream classes differs by ``role``.

    The check is scoped to the ``register(...)`` calls — the thirteen names published under both
    ``keras.losses`` and ``keras.metrics`` — and a reload is how it is exercised, because a
    clobber is only observable when the same name is registered twice.

    ``SparseF1Score`` is excluded, and that exclusion is about the RELOAD rather than about the
    class: confluid stays silent when the identical class OBJECT is re-registered
    (``registry._store``: ``existing.cls is entry.cls``), which is what makes re-importing this
    module free for the upstream classes — ``keras.metrics.F1Score`` is the same object every
    time. A ``@configurable`` class defined HERE is rebuilt by the reload, so the registry sees
    a genuinely different object under the same name and tags and correctly says so. Nothing in
    a normal run reloads a module, so the warning cannot occur outside this test.
    """
    import confluid.registry as registry_module

    collected: list = []
    original = registry_module.logger
    registry_module.logger = SimpleNamespace(  # type: ignore[assignment]
        warning=collected.append, debug=lambda m: None, info=lambda m: None, trace=lambda m: None
    )
    try:
        importlib.reload(kc)
    finally:
        registry_module.logger = original  # type: ignore[assignment]
    upstream = [message for message in collected if "SparseF1Score" not in str(message)]
    assert upstream == []


def test_the_sparse_f1_shim_registers_as_a_keras_classification_metric() -> None:
    """The one sonair-owned class in this module is discoverable like every registered one."""
    found = get_registry().get_class("SparseF1Score", task="classification", role="metric", framework="keras")
    assert found is kc.SparseF1Score
    assert "SparseF1Score" in kc.__all__
    # Zero-arg construction, and the default name is the column the sweep compares on.
    assert kc.SparseF1Score().name == "f1_macro"


def test_registered_classes_are_the_upstream_keras_ones() -> None:
    """No wrappers: the registered objects ARE ``keras.losses`` / ``keras.metrics`` classes."""
    for cls in kc._CLASSIFICATION_LOSSES:
        assert getattr(keras.losses, cls.__name__) is cls
    for cls in kc._CLASSIFICATION_METRICS:
        assert getattr(keras.metrics, cls.__name__) is cls
    assert all(c.__module__.startswith("keras") for c in (*kc._CLASSIFICATION_LOSSES, *kc._CLASSIFICATION_METRICS))


def test_registry_resolves_a_keras_loss_by_name() -> None:
    """The picker's other half: a chosen name must build.

    ``role="loss"`` is required here because the same name is also a metric — which is the
    feature, not an inconvenience: the picker always knows which slot it is filling.
    """
    cls = get_registry().get_class("SparseCategoricalCrossentropy", role="loss")
    assert cls is not None, "registered under its class name — a rename would break every config naming it"
    loss = cls(from_logits=True)
    assert isinstance(loss, keras.losses.Loss)
    assert loss.from_logits is True


# ---------------------------------------------------------------------------
# the wiring guidance in the module docstring, held to the numbers
# ---------------------------------------------------------------------------


def test_every_registered_loss_takes_logits_or_raw_scores() -> None:
    """The crossentropy family must expose ``from_logits`` — the head emits raw logits.

    Registering a crossentropy loss whose only spelling assumed probabilities would hand
    operators the exact failure `_resolve_task_defaults` documents: finite, clipped, wrong.
    """
    import inspect

    for cls in kc._CLASSIFICATION_LOSSES:
        if "Crossentropy" in cls.__name__:
            assert "from_logits" in inspect.signature(cls.__init__).parameters, cls.__name__


@pytest.mark.parametrize(
    "name, kwargs, sparse",
    [
        ("SparseCategoricalAccuracy", {}, True),
        ("SparseTopKCategoricalAccuracy", {"k": 2}, True),
        ("CategoricalAccuracy", {}, False),
        ("TopKCategoricalAccuracy", {"k": 2}, False),
        ("F1Score", {"average": "macro"}, False),  # threshold=None -> argmax
    ],
)
def test_argmax_based_metrics_read_the_same_on_logits_and_probabilities(name: str, kwargs: dict, sparse: bool) -> None:
    """Order-preserving metrics need no configuration on a logits head."""
    cls = getattr(keras.metrics, name)
    y_true = IDS if sparse else TARGETS
    assert _value(cls(**kwargs), LOGITS, y_true) == _value(cls(**kwargs), PROBS, y_true)


@pytest.mark.parametrize("name", ["Precision", "Recall", "TruePositives", "FalsePositives"])
def test_threshold_metrics_need_zero_on_a_logits_head(name: str) -> None:
    """``thresholds=0.0`` is the logit-space equivalent of the probability default 0.5."""
    cls = getattr(keras.metrics, name)
    assert _value(cls(thresholds=0.0), LOGITS) == _value(cls(), PROBS)
    # ...and the default is NOT equivalent, which is why the docstring spells it out.
    assert _value(cls(), LOGITS) != _value(cls(), PROBS)


def test_the_counter_metrics_need_a_multi_hot_target_not_sparse_ids() -> None:
    """Which metrics apply is decided by the TARGET SHAPE before the logits question.

    ``Precision`` and friends compare element-wise against ``y_pred``, so they need a float
    ``[N, C]`` target — the multi-hot a ``recordstream.MultiLabel`` produces. Handed the int64
    ids ``[N]`` of a single-label run they raise on the broadcast, which is what
    ``sonair train --framework keras`` did with them wired next to ``SparseCategoricalAccuracy``.
    """
    assert _value(keras.metrics.Precision(thresholds=0.0), LOGITS, TARGETS) > 0  # multi-hot: fine
    with pytest.raises(Exception):
        _value(keras.metrics.Precision(thresholds=0.0), LOGITS, IDS)  # sparse ids: shape mismatch


def test_binary_accuracy_uses_the_singular_threshold_knob() -> None:
    assert _value(keras.metrics.BinaryAccuracy(threshold=0.0), LOGITS) == _value(keras.metrics.BinaryAccuracy(), PROBS)


def test_auc_is_the_metric_that_carries_from_logits() -> None:
    kw = dict(multi_label=True, num_labels=4)
    assert _value(keras.metrics.AUC(from_logits=True, **kw), LOGITS) == _value(keras.metrics.AUC(**kw), PROBS)


def test_fscore_rejects_a_logit_space_threshold() -> None:
    """Why the docstring says to leave ``threshold`` unset rather than pass 0.0."""
    with pytest.raises(ValueError):
        keras.metrics.F1Score(threshold=0.0)


def test_metrics_that_cannot_consume_logits_are_not_registered() -> None:
    """The exclusions, with the reason each was excluded reproduced.

    ``Accuracy`` wants already-argmaxed ids; the ``*At*`` family sweeps thresholds across
    ``[0, 1]`` with no escape hatch, so it silently reports a different number on logits.
    """
    registered = {c.__name__ for c in kc._CLASSIFICATION_METRICS}
    assert "Accuracy" not in registered
    assert _value(keras.metrics.Accuracy(), LOGITS) == 0.0

    for name, kwargs in [
        ("PrecisionAtRecall", {"recall": 0.5}),
        ("RecallAtPrecision", {"precision": 0.5}),
        ("SensitivityAtSpecificity", {"specificity": 0.5}),
        ("SpecificityAtSensitivity", {"sensitivity": 0.5}),
    ]:
        assert name not in registered
    cls = keras.metrics.SensitivityAtSpecificity
    assert _value(cls(specificity=0.5), LOGITS) != _value(cls(specificity=0.5), PROBS)


# ---------------------------------------------------------------------------
# wiring: a registered class must survive the trip from YAML into compile()
# ---------------------------------------------------------------------------


def _materialize(yaml_text: str, tmp_path: Path) -> KerasClassifier:
    """Build a `runnable:` block the way `sonair train --framework keras` does (against its document)."""
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml_text)
    document = confluid.load(str(path), until="raw")
    return cast(KerasClassifier, confluid.load(document["runnable"], context=document))


_WIRED = """
runnable: !class:sonair.KerasClassifier
  model: !lazy:sonair.KerasModel
    model_name: MobileNetV3Small
    input_shape: [32, 32, 3]
  loss: !class:keras.losses.SparseCategoricalCrossentropy
    from_logits: true
  metrics:
    - !class:keras.metrics.SparseCategoricalAccuracy
    - !class:keras.metrics.Precision
      thresholds: 0.0
num_classes: 3
"""


def test_a_yaml_wired_loss_and_metric_reach_compile_built(tmp_path: Path) -> None:
    """The registrations only pay off if a chosen class survives into ``compile()``.

    ``loss`` is declared ``Partial[KerasLoss]``, so the slot keeps its marker at load whatever
    the value spells — deferral is the receiver's declared contract. ``_compile()`` is what
    flows it, and ``flow()`` is idempotent, so that one unconditional line is equally correct
    for a marker and for a live object an operator assigned directly.
    """
    clf = _materialize(_WIRED, tmp_path)
    assert isinstance(clf.loss, Fluid)  # a `Partial[T]` slot stays deferred until `_compile()`
    clf._compile()

    # `clf.model` IS the compiled `keras.Model` — `KerasModel` subclasses it, so `_compile()`
    # compiles the wired object itself and its Keras-owned state reads back off it directly.
    network = clf.model
    assert isinstance(network.loss, keras.losses.SparseCategoricalCrossentropy)  # ...live at compile
    assert network.loss.from_logits is True
    kinds = {type(m).__name__: m for m in network._compile_metrics._user_metrics}
    assert "SparseCategoricalAccuracy" in kinds
    assert kinds["Precision"].thresholds == [0.0]


def test_the_instance_form_with_parens_works_too(tmp_path: Path) -> None:
    """The trailing ``()`` is INERT — both spellings must reach the same place.

    Confluid merged its eager and deferred markers (2026-08-11), so ``X()`` and ``X`` are the
    same target: neither builds here, because the ``Partial[KerasLoss]`` slot defers it either way.
    This pins that the parens spelling did not survive as a second construction mode.
    """
    clf = _materialize(_WIRED.replace("SparseCategoricalCrossentropy\n", "SparseCategoricalCrossentropy()\n"), tmp_path)
    assert isinstance(clf.loss, Fluid)
    clf._compile()
    assert isinstance(clf.model.loss, keras.losses.SparseCategoricalCrossentropy)
    assert clf.model.loss.from_logits is True


def test_an_explicitly_wired_loss_is_never_replaced_by_the_sentinel(tmp_path: Path) -> None:
    """`_resolve_task_defaults` fills an UNSET slot only — a chosen loss survives it."""
    clf = _materialize(_WIRED, tmp_path)
    clf.multilabel = True  # a sentinel would now pick BinaryCrossentropy
    clf._compile()
    assert isinstance(clf.model.loss, keras.losses.SparseCategoricalCrossentropy)


def test_a_registered_loss_and_metric_drive_a_real_compile_and_fit() -> None:
    """End to end: what a config wiring these actually does."""
    model = keras.Sequential([keras.layers.Input((4,)), keras.layers.Dense(3)])  # bare Dense -> logits
    model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    x = _RNG.normal(size=(32, 4)).astype("float32")
    y = _RNG.integers(0, 3, size=32).astype("int64")
    history = model.fit(x, y, epochs=1, batch_size=8, verbose=0)
    assert "sparse_categorical_accuracy" in history.history
    assert np.isfinite(history.history["loss"][0])


def test_the_sparse_f1_shim_agrees_with_sklearn_on_INT_ID_targets() -> None:
    """What upstream's `F1Score` cannot do: read the `[N]` int ids a single-label run produces.

    Accumulated across batches, exactly as `compile()` would drive it, and compared against the
    reduction the torch backends' `MulticlassF1Score(average="macro")` computes — that agreement
    is what puts both engines' numbers in ONE column of a cross-framework comparison.
    """
    sklearn_metrics = pytest.importorskip("sklearn.metrics")

    rng = np.random.default_rng(11)
    targets = rng.integers(0, 10, 320)
    predicted = np.where(rng.random(320) < 0.6, targets, rng.integers(0, 10, 320))
    logits = (np.eye(10)[predicted] * 4.0 + rng.normal(0, 0.1, (320, 10))).astype("float32")

    metric = kc.SparseF1Score()
    for start in range(0, 320, 32):
        metric.update_state(targets[start : start + 32].astype("int64"), logits[start : start + 32])

    expected = sklearn_metrics.f1_score(targets, logits.argmax(-1), average="macro", labels=range(10), zero_division=0)
    assert float(metric.result()) == pytest.approx(expected, abs=1e-5)


def test_the_sparse_f1_shim_reads_RAW_LOGITS() -> None:
    """`threshold=None` argmaxes, and argmax is invariant under softmax — so a bare `Dense`
    head needs no `from_logits` knob, the same property that makes `SparseCategoricalAccuracy`
    safe here."""
    rng = np.random.default_rng(12)
    targets = rng.integers(0, 4, 64)
    logits = (np.eye(4)[targets] * 3.0 + rng.normal(0, 0.5, (64, 4))).astype("float32")
    probabilities = (np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)).astype("float32")

    from_logits, from_probabilities = kc.SparseF1Score(), kc.SparseF1Score()
    from_logits.update_state(targets.astype("int64"), logits)
    from_probabilities.update_state(targets.astype("int64"), probabilities)

    assert float(from_logits.result()) == pytest.approx(float(from_probabilities.result()), abs=1e-6)


def test_the_sparse_f1_shims_name_follows_its_averaging() -> None:
    """Derived rather than a literal default, so `average="micro"` cannot be recorded under the
    macro name — one name per quantity, decided where the metric is wired."""
    assert kc.SparseF1Score().name == "f1_macro"
    assert kc.SparseF1Score(average="micro").name == "f1_micro"
    assert kc.SparseF1Score(name="whatever_the_config_says").name == "whatever_the_config_says"
