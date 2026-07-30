"""The Keras loss/metric registrations — the ``framework="keras"`` half of the picker.

These pin three things a broken registration would silently cost:

1. every class in ``_CLASSIFICATION_LOSSES`` / ``_CLASSIFICATION_METRICS`` is discoverable
   under ``task="classification"`` + its role + ``framework="keras"`` (a dropped tag empties
   a picker rather than failing anything);
2. ``__all__`` names every one of them, because ``scan_module`` filters by ``__module__``
   (``keras.src.*`` here) and the ``__all__``-gated pass is what surfaces them to StreamStudio;
3. no registered NAME is claimed twice — confluid's registry is keyed by class name, and
   Keras publishes thirteen names under both ``keras.losses`` and ``keras.metrics``, so
   registering a metric duplicate would silently overwrite its loss.

Plus the numeric facts the module docstring's wiring guidance rests on: which metrics are
safe on the classifier's raw-logits head, and what to set for the ones that are not.
"""

from pathlib import Path
from typing import cast

import confluid
import numpy as np
import pytest
from confluid import Fluid, get_registry

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
    names = get_registry().list_classes(task="classification", role="loss", framework="keras")
    for cls in kc._CLASSIFICATION_LOSSES:
        assert cls.__name__ in names, f"{cls.__name__} not registered as a keras classification loss"


def test_every_keras_metric_is_registered_under_its_framework() -> None:
    registry = get_registry()
    names = registry.list_classes(task="classification", role="metric", framework="keras")
    for cls in kc._CLASSIFICATION_METRICS:
        assert cls.__name__ in names, f"{cls.__name__} not registered as a keras classification metric"
    # category is the derived `classification_metric`, which navigaitor's metric-slot picker reads.
    assert "SparseCategoricalAccuracy" in registry.list_classes(category="classification_metric")


def test_all_exports_registered_losses_and_metrics() -> None:
    """``__all__`` must name every registered class or StreamStudio's pass never sees it."""
    for cls in (*kc._CLASSIFICATION_LOSSES, *kc._CLASSIFICATION_METRICS):
        assert cls.__name__ in kc.__all__, f"{cls.__name__} registered but missing from __all__"
    assert len(kc.__all__) == len(set(kc.__all__))


def test_no_name_is_registered_twice() -> None:
    """confluid's registry is keyed by NAME, so a loss/metric name clash would overwrite silently.

    Keras publishes thirteen names under both ``keras.losses`` and ``keras.metrics``
    (``BinaryCrossentropy``, ``Hinge``, ``KLDivergence``, …). This is why the metric set omits
    the crossentropy/hinge duplicates.
    """
    loss_names = {c.__name__ for c in kc._CLASSIFICATION_LOSSES}
    metric_names = {c.__name__ for c in kc._CLASSIFICATION_METRICS}
    assert not (loss_names & metric_names)
    shared = {n for n in dir(keras.losses) if n[0].isupper()} & {n for n in dir(keras.metrics) if n[0].isupper()}
    assert shared, "expected keras to publish some names in both namespaces (the constraint being pinned)"
    assert not (metric_names & shared), f"metric duplicates of a loss name would overwrite it: {metric_names & shared}"


def test_registered_classes_are_the_upstream_keras_ones() -> None:
    """No wrappers: the registered objects ARE ``keras.losses`` / ``keras.metrics`` classes."""
    for cls in kc._CLASSIFICATION_LOSSES:
        assert getattr(keras.losses, cls.__name__) is cls
    for cls in kc._CLASSIFICATION_METRICS:
        assert getattr(keras.metrics, cls.__name__) is cls
    assert all(c.__module__.startswith("keras") for c in (*kc._CLASSIFICATION_LOSSES, *kc._CLASSIFICATION_METRICS))


def test_registry_resolves_a_keras_loss_by_name() -> None:
    """The picker's other half: a chosen name must build."""
    cls = get_registry().get_class("SparseCategoricalCrossentropy")
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
    ``sonair keras train`` did with them wired next to ``SparseCategoricalAccuracy``.
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
    """Build a `runnable:` block the way `sonair keras train` does (against its document)."""
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml_text)
    document = confluid.load_config(str(path))
    return cast(KerasClassifier, confluid.materialize(document["runnable"], context=document))


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

    A nested ``!class:`` in a CONSTRUCTOR slot stays a deferred `Class` marker (only the
    instance form ``!class:X()`` builds eagerly), so the slots are flowed at USE — the same
    thing ``_callbacks()`` does for the ``logger`` slot. Without it Keras took the unbuilt
    marker for a custom callable and failed inside ``compile``.
    """
    clf = _materialize(_WIRED, tmp_path)
    assert isinstance(clf.loss, Fluid)  # deferred as materialized...
    clf._compile()

    network = clf._network
    assert isinstance(network.loss, keras.losses.SparseCategoricalCrossentropy)  # ...live at compile
    assert network.loss.from_logits is True
    kinds = {type(m).__name__: m for m in network._compile_metrics._user_metrics}
    assert "SparseCategoricalAccuracy" in kinds
    assert kinds["Precision"].thresholds == [0.0]


def test_the_instance_form_with_parens_works_too(tmp_path: Path) -> None:
    """`!class:X()` builds at materialize time — both spellings must reach the same place."""
    clf = _materialize(_WIRED.replace("SparseCategoricalCrossentropy\n", "SparseCategoricalCrossentropy()\n"), tmp_path)
    assert isinstance(clf.loss, keras.losses.SparseCategoricalCrossentropy)
    clf._compile()
    assert clf._network.loss.from_logits is True


def test_an_explicitly_wired_loss_is_never_replaced_by_the_sentinel(tmp_path: Path) -> None:
    """`_resolve_task_defaults` fills an UNSET slot only — a chosen loss survives it."""
    clf = _materialize(_WIRED, tmp_path)
    clf.multilabel = True  # a sentinel would now pick BinaryCrossentropy
    clf._compile()
    assert isinstance(clf._network.loss, keras.losses.SparseCategoricalCrossentropy)


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
