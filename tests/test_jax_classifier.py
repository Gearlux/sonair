# mypy: disable-error-code="attr-defined"
"""``JaxClassifier`` — the jitted optax loop, and ``FlaxConvNet``.

Two claims worth pinning. The first is the usual one: everything shared (labels, class names,
the batch shape, the prediction contract, the artifacts) comes from recordstream and marainer
rather than being reimplemented for JAX. The second is specific to this backend — it owns its
loop, so the things a `fit()` would have done for it (the per-epoch reshuffle, building the
optimizer once) have to be done here, and forgetting them is invisible rather than loud.

Skipped when jax is not installed — it is a selectable backend (``sonair[jax]``), not a core
dependency.
"""

from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pytest

pytest.importorskip("jax", reason="the jax backend needs sonair[jax]")

import jax  # noqa: E402 - after the skip
import jax.numpy as jnp  # noqa: E402
from confluid import Fluid, LazyClass, lazy_param_names, output_specs  # noqa: E402
from flax import nnx  # noqa: E402
from recordstream import Stream  # noqa: E402
from recordstream.items import Image, Label, MultiLabel  # noqa: E402

from sonair.convnet.flax_convnet import FlaxConvNet  # noqa: E402
from sonair.convnet.spec import trainable_parameter_count  # noqa: E402
from sonair.jax_classifier import JaxClassifier, _as_loss, _as_metric, _as_optimizer  # noqa: E402

INPUT = (16, 16, 3)


class _CaptureSink:
    def __init__(self) -> None:
        self.rows: List[Tuple[Any, Any]] = []
        self.closed = False

    def write(self, prediction: Any, metadata: Any) -> None:
        self.rows.append((prediction, metadata))

    def close(self) -> None:
        self.closed = True


def _records(n: int = 12, labels: str = "ids") -> list:
    """Channels-LAST records — what a JAX/Keras preprocessing chain leaves behind."""
    names = ["cat", "dog", "bird"]
    rng = np.random.default_rng(0)
    return [
        {
            "image": Image(rng.random(INPUT).astype("float32"), layout="HWC"),
            "class": Label(names[i % 3] if labels == "names" else i % 3),
            "row_id": i,
        }
        for i in range(n)
    ]


def _classifier(**kw: Any) -> JaxClassifier:
    defaults: dict = dict(
        model=LazyClass(FlaxConvNet, input_shape=INPUT),
        max_epochs=1,
        batch_size=4,
        experiment_name="test",
        run_name="r",
    )
    defaults.update(kw)
    return JaxClassifier(**defaults)


# --------------------------------------------------------------------------- #
# FlaxConvNet — the shared spec, in nnx
# --------------------------------------------------------------------------- #


def test_flax_convnet_matches_the_shared_spec() -> None:
    """The comparison this whole architecture exists for: the count is DERIVED from the spec,
    not measured from one framework, so each implementation is checked against the claim."""
    model = FlaxConvNet(num_classes=10, input_shape=INPUT)
    model.solidify()  # lazy by mandate: no parameters exist until the build
    params = sum(int(v.size) for v in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param)))

    assert params == trainable_parameter_count(10)


def test_flax_convnet_is_an_nnx_module() -> None:
    """Same decision as `KerasModel` being a `keras.Model`: the classifier drives the slot
    without unwrapping, and `nnx.state` / `nnx.Optimizer` reach through the wrapper."""
    assert issubclass(FlaxConvNet, nnx.Module)


def test_zero_arg_construction_builds_nothing() -> None:
    """The workspace lazy-init mandate — and the reason `nnx.data` is needed at build time."""
    model = FlaxConvNet()

    assert model._model is None
    with pytest.raises(ValueError, match="num_classes"):
        model.solidify()


def test_a_never_flowed_model_builds_on_first_use() -> None:
    model = FlaxConvNet(num_classes=3, input_shape=INPUT)

    assert model(jnp.ones((2, *INPUT))).shape == (2, 3)


def test_the_head_emits_raw_logits() -> None:
    """No activation on the head, like every other backend — so `from_logits` losses apply.

    A softmax added later would make the rows sum to 1; this would catch that.
    """
    out = np.asarray(FlaxConvNet(num_classes=4, input_shape=INPUT)(jnp.ones((2, *INPUT))))

    assert not np.allclose(out.sum(axis=-1), 1.0)


def test_the_seed_makes_init_reproducible() -> None:
    """JAX makes randomness explicit, so the same config gives the same weights."""
    a = np.asarray(FlaxConvNet(num_classes=3, input_shape=INPUT, seed=7)(jnp.ones((1, *INPUT))))
    b = np.asarray(FlaxConvNet(num_classes=3, input_shape=INPUT, seed=7)(jnp.ones((1, *INPUT))))
    c = np.asarray(FlaxConvNet(num_classes=3, input_shape=INPUT, seed=8)(jnp.ones((1, *INPUT))))

    assert np.allclose(a, b)
    assert not np.allclose(a, c)


# --------------------------------------------------------------------------- #
# The batch shape
# --------------------------------------------------------------------------- #


def test_batches_are_numpy_channels_last() -> None:
    """recordstream hands over NUMPY, which is exactly what `jnp.asarray` takes — that is why
    this backend needs no conversion layer, and why those return types are not torch-typed."""
    x, y = _classifier()._sequence(Stream(source=_records(8)))[0]

    assert isinstance(x, np.ndarray) and x.shape == (4, *INPUT) and x.dtype == np.float32
    assert isinstance(y, np.ndarray) and y.shape == (4,) and y.dtype == np.int64


def test_a_multilabel_target_becomes_a_multi_hot_matrix() -> None:
    records = [{"image": Image(np.zeros(INPUT, "float32"), layout="HWC"), "class": MultiLabel([0, 2])}]
    clf = _classifier(num_classes=3)
    clf.multilabel = True

    assert clf._sequence(Stream(source=records))[0][1].tolist() == [[1.0, 0.0, 1.0]]


def test_predict_mode_yields_inputs_only() -> None:
    assert len(_classifier()._sequence(Stream(source=_records(4)), with_targets=False)[0]) == 1


# --------------------------------------------------------------------------- #
# Name resolution — optax has no classes to tag
# --------------------------------------------------------------------------- #


def test_losses_and_optimizers_resolve_by_name() -> None:
    """optax ships FUNCTIONS and FACTORIES, not classes, so there is nothing for confluid's
    registry to tag — the same situation the fastai backend's metrics are in."""
    import optax

    assert (
        _as_loss("softmax_cross_entropy_with_integer_labels") is optax.losses.softmax_cross_entropy_with_integer_labels
    )
    assert _as_metric("accuracy").__name__ == "_accuracy"

    tx = _as_optimizer("adam", 1e-3)
    assert hasattr(tx, "update") and hasattr(tx, "init"), "an optax GradientTransformation"


def test_an_unknown_name_says_where_names_come_from() -> None:
    with pytest.raises(ValueError, match="optax.losses"):
        _as_loss("nope")
    with pytest.raises(ValueError, match="optax"):
        _as_optimizer("nope", 1e-3)
    with pytest.raises(ValueError, match="Known names"):
        _as_metric("nope")


def test_the_optimizer_name_is_called_with_the_shared_learning_rate() -> None:
    """optax optimizers are factories, so the `learning_rate` knob stays the ONE place the LR
    is written — shared with every other backend's config."""
    import optax

    assert _as_optimizer("sgd", 0.5) is not optax.sgd  # it was CALLED, not handed back


# --------------------------------------------------------------------------- #
# Task-shaped defaults
# --------------------------------------------------------------------------- #


def test_defaults_are_resolved_from_the_data_not_the_signature() -> None:
    clf = _classifier(train_set=Stream(source=_records()))
    assert clf.loss is None and clf.metrics is None

    clf.initialize()
    assert clf.loss == "softmax_cross_entropy_with_integer_labels"
    assert clf.metrics == ["accuracy"]


def test_a_multilabel_run_picks_the_sigmoid_pair() -> None:
    """The JAX spelling of BCEWithLogitsLoss, against the float multi-hot `_to_arrays` makes."""
    clf = _classifier()
    clf.multilabel = True
    clf._resolve_task_defaults()

    assert clf.loss == "sigmoid_binary_cross_entropy"
    assert clf.metrics == ["accuracy_multi"]


def test_an_explicitly_wired_loss_is_never_replaced() -> None:
    clf = _classifier(loss="softmax_cross_entropy_with_integer_labels")
    clf.multilabel = True
    clf._resolve_task_defaults()

    assert clf.loss == "softmax_cross_entropy_with_integer_labels"


# --------------------------------------------------------------------------- #
# The loop — what a `fit()` would have done for us
# --------------------------------------------------------------------------- #


def test_fit_trains_and_reports(tmp_path: Path) -> None:
    clf = _classifier(
        train_set=Stream(source=_records(12)),
        val_set=Stream(source=_records(8)),
        max_epochs=2,
    )
    clf.metrics_sidecar_path = str(tmp_path / "metrics.json")
    clf.run()

    assert "loss" in clf.computed_metrics and "val_accuracy" in clf.computed_metrics
    assert (tmp_path / "metrics.json").exists()
    assert clf.num_classes == 3


def test_the_per_epoch_reshuffle_happens() -> None:
    """A `fit()` calls this between epochs; a hand-written loop must. Forgetting it is
    INVISIBLE — training silently runs on one fixed row order forever."""
    clf = _classifier(train_set=Stream(source=_records(12)), max_epochs=3)
    seen: List[int] = []
    real = type(clf)._sequence

    def spy(self: Any, source: Any, shuffle: bool = False, with_targets: bool = True) -> Any:
        sequence = real(self, source, shuffle=shuffle, with_targets=with_targets)
        original = sequence.on_epoch_end

        def counted() -> None:
            seen.append(1)
            original()

        sequence.on_epoch_end = counted  # type: ignore[method-assign]
        return sequence

    type(clf)._sequence = spy  # type: ignore[method-assign]
    try:
        clf.run()
    finally:
        type(clf)._sequence = real  # type: ignore[method-assign]

    assert len(seen) == 3, "one reshuffle per epoch"


def test_jit_is_a_knob_and_both_paths_agree() -> None:
    """`jit` off exists so a step's traceback is readable; the numbers must not change."""
    jitted = _classifier(train_set=Stream(source=_records(8)), jit=True)
    eager = _classifier(train_set=Stream(source=_records(8)), jit=False)
    jitted.run()
    eager.run()

    assert jitted.computed_metrics["loss"] == pytest.approx(eager.computed_metrics["loss"], rel=1e-4)


def test_string_targets_are_encoded_through_a_fitted_label_map() -> None:
    clf = _classifier(train_set=Stream(source=_records(9, labels="names")))
    clf.run()

    assert clf.num_classes == 3
    assert clf.class_names == ["bird", "cat", "dog"]


# --------------------------------------------------------------------------- #
# Predictions + outputs
# --------------------------------------------------------------------------- #


def test_predict_writes_one_entry_per_record_with_its_metadata() -> None:
    sink = _CaptureSink()
    clf = _classifier(
        model=LazyClass(FlaxConvNet, input_shape=INPUT, num_classes=3),
        test_set=Stream(source=_records(7)),
        task="predict",
        batch_size=3,
    )
    clf.predictions_sink = sink  # type: ignore[assignment]
    clf.run()

    assert len(sink.rows) == 7, "a short final batch must not be dropped"
    prediction, metadata = sink.rows[0]
    assert set(prediction) == {"logits", "probs", "class_idx"}
    assert metadata == {"row_id": 0}
    assert pytest.approx(float(prediction["probs"].sum()), abs=1e-5) == 1.0
    assert sink.closed


def test_outputs_are_declared_and_raise_before_the_run() -> None:
    names = {spec["name"] for spec in output_specs(JaxClassifier)}
    assert {"trained_model", "computed_metrics", "class_names"} <= names

    clf = _classifier()
    assert isinstance(clf.model, Fluid)
    with pytest.raises(RuntimeError, match="still deferred"):
        _ = clf.trained_model


def test_discovery_tags_and_lazy_slots() -> None:
    assert JaxClassifier.__confluid_task__ == "classification"
    assert JaxClassifier.__confluid_framework__ == "jax"
    assert {"model", "train_set", "test_set"} <= lazy_param_names(JaxClassifier)


@pytest.mark.parametrize("method", ["test", "predict"])
def test_eval_verbs_without_a_test_set_are_clear_errors(method: str) -> None:
    clf = _classifier(model=FlaxConvNet(num_classes=3, input_shape=INPUT), task=method)
    with pytest.raises(ValueError, match="test_set"):
        getattr(clf, method)()


def test_it_registers_no_loss_module() -> None:
    """optax losses are functions, so a config NAMES them — there is nothing to register."""
    import importlib.util

    from sonair.cli import IMPLEMENTED

    assert IMPLEMENTED["jax"].registrations is None
    assert importlib.util.find_spec("sonair.jax_classification") is None
