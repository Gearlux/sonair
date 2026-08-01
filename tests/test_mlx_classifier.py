# mypy: disable-error-code="attr-defined"
"""``MlxClassifier`` — training on Apple-silicon unified memory, and ``MlxConvNet``.

Two things are worth pinning beyond the usual reuse claims. MLX arrays are **lazy**, so the
loop's `mx.eval` calls are load-bearing rather than defensive — omitting one does not raise, it
grows the graph until memory runs out. And there is **no device transfer** on unified memory,
which is the one place sonair's batch path does genuinely less work than the other backends
rather than the same work differently.

Skipped where MLX is absent — it is Apple-silicon only (`sys_platform == 'darwin' and
platform_machine == 'arm64'` in the extra), so CI on Linux can never run this. The SOURCE-only
tests below still run everywhere, deliberately: the machines that can break the platform
guard or drop a class from the CLI are exactly the ones without MLX.
"""

import importlib.util
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pytest

requires_mlx = pytest.mark.skipif(
    importlib.util.find_spec("mlx") is None,
    reason="the mlx backend needs sonair[mlx] (Apple silicon only)",
)

INPUT = (16, 16, 3)


# --------------------------------------------------------------------------- #
# Source-only — these must keep running where MLX is ABSENT
# --------------------------------------------------------------------------- #


def test_the_extra_carries_the_platform_marker() -> None:
    """`pip install sonair[mlx]` must not fail outright off Apple silicon — the marker makes
    the extra installable everywhere and empty where MLX cannot run."""
    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()

    assert "sys_platform == 'darwin' and platform_machine == 'arm64'" in pyproject


def test_mlx_is_an_implemented_backend_not_a_template() -> None:
    from sonair.cli import IMPLEMENTED, TEMPLATES

    assert "mlx" in IMPLEMENTED
    assert "mlx" not in TEMPLATES
    assert IMPLEMENTED["mlx"].registrations is None, "mlx.nn.losses are functions — nothing to register"


def test_every_advertised_backend_is_implemented() -> None:
    """The end of the roadmap this project shipped with: nothing advertised is a stub.

    The COUNT is deliberately not asserted here — `sonair torch` landed as an eighth backend
    after this file was written, and a hard number turns every future backend into a failure in
    the mlx suite. `tests/test_cli_frameworks.py::test_the_implemented_set_is_the_one_we_expect`
    owns the exact roster; this one owns the invariant that `TEMPLATES` has been emptied.
    """
    from sonair.cli import IMPLEMENTED, TEMPLATES

    assert TEMPLATES == {}
    assert "mlx" in IMPLEMENTED


# --------------------------------------------------------------------------- #
# MlxConvNet — the fourth implementation of the shared spec
# --------------------------------------------------------------------------- #


@requires_mlx
def test_it_matches_the_shared_spec() -> None:
    from sonair.convnet.mlx_convnet import MlxConvNet
    from sonair.convnet.spec import trainable_parameter_count

    model = MlxConvNet(num_classes=10, input_shape=INPUT)
    model.solidify()

    assert model.parameter_count == trainable_parameter_count(10)


@requires_mlx
def test_batchnorm_running_stats_are_excluded_from_the_count() -> None:
    """MLX keeps them in the SAME parameter tree as the weights, unlike the other three, which
    hold them as separate non-trainable state. Counting them would break the four-way match."""
    from mlx.utils import tree_flatten

    from sonair.convnet.mlx_convnet import MlxConvNet

    model = MlxConvNet(num_classes=10, input_shape=INPUT)
    model.solidify()
    leaves: Any = tree_flatten(model.model.parameters())
    paths = [p for p, _ in leaves]

    assert any(p.endswith("running_mean") for p in paths), "they ARE in the tree — hence the filter"
    assert model.parameter_count < sum(v.size for _, v in leaves)


@requires_mlx
def test_zero_arg_construction_allocates_nothing() -> None:
    from sonair.convnet.mlx_convnet import MlxConvNet

    model = MlxConvNet()

    # Asserted through `parameters()` rather than the backing attribute: this class IS an
    # `nn.Module`, so "allocates nothing" is a statement about the parameter tree the optimizer
    # would see, and that survives a rename of where the network is stashed.
    assert model.net is None
    assert not model.parameters()
    with pytest.raises(ValueError, match="num_classes"):
        model.solidify()


@requires_mlx
def test_it_is_an_mlx_module() -> None:
    """The twin of `FlaxConvNet` being an `nnx.Module` and `TimmModel` an `nn.Module`.

    It is what lets `MlxClassifier` hand the slot value straight to `nn.value_and_grad` /
    `optimizer.update` instead of reaching through to `self.model.model`.
    """
    import mlx.nn as nn

    from sonair.convnet.mlx_convnet import MlxConvNet

    assert issubclass(MlxConvNet, nn.Module)


@requires_mlx
def test_the_built_network_is_visible_to_parameters() -> None:
    """The underscore trap, pinned.

    `nn.Module.valid_parameter_filter` ends in `and not key.startswith("_")`, so holding the
    network in `self._model` — the name the Keras / TensorFlow / Flax twins use — hides it from
    `parameters()` entirely. The optimizer would then see an empty tree and the loop would apply
    gradients to nothing forever, silently. Nothing else in the suite would catch that: the
    forward pass still works, the loss is still finite, it just never moves.
    """
    from sonair.convnet.mlx_convnet import MlxConvNet

    model = MlxConvNet(num_classes=10, input_shape=INPUT)
    model.solidify()

    assert model.parameters(), "the network must be reachable through the module's own tree"
    assert all(not key.startswith("_") for key in model.parameters())
    assert model.trainable_parameters()


@requires_mlx
def test_calling_it_does_not_reset_the_training_mode() -> None:
    """`training` defaults to None — *leave the mode alone* — and that is not cosmetic.

    MLX carries train/eval as module STATE. `nn.value_and_grad` calls `model(x)` with no flag,
    so a `training: bool = False` default (what the Keras and Flax twins take) would flip the
    network to eval on every step of the training loop and dropout / batch-norm would never
    train. The failure is silent: the loss stays finite and merely wrong.
    """
    import mlx.core as mx

    from sonair.convnet.mlx_convnet import MlxConvNet

    model = MlxConvNet(num_classes=4, input_shape=INPUT)
    model.train(True)
    model(mx.ones((2, *INPUT)))

    assert model.training is True
    # ...and an explicit flag still sets it, which is what the eval paths rely on.
    model(mx.ones((2, *INPUT)), training=False)
    assert model.training is False


@requires_mlx
def test_the_head_emits_raw_logits() -> None:
    import mlx.core as mx

    from sonair.convnet.mlx_convnet import MlxConvNet

    out = np.asarray(MlxConvNet(num_classes=4, input_shape=INPUT)(mx.ones((2, *INPUT))))

    assert not np.allclose(out.sum(axis=-1), 1.0)


# --------------------------------------------------------------------------- #
# The batch shape
# --------------------------------------------------------------------------- #


def _records(n: int = 12, labels: str = "ids") -> list:
    from recordstream.items import Image, Label

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


def _classifier(**kw: Any) -> Any:
    from confluid import LazyClass

    from sonair.convnet.mlx_convnet import MlxConvNet
    from sonair.mlx_classifier import MlxClassifier

    defaults: dict = dict(
        model=LazyClass(MlxConvNet, input_shape=INPUT),
        max_epochs=1,
        batch_size=4,
        experiment_name="test",
        run_name="r",
    )
    defaults.update(kw)
    return MlxClassifier(**defaults)


@requires_mlx
def test_targets_are_int32_not_int64() -> None:
    """MLX indexes with int32 and `cross_entropy` expects it — the other backends name int64
    for exactly the mirror-image reason. Named explicitly rather than inherited from the data."""
    from recordstream import Stream

    _, y = _classifier()._sequence(Stream(source=_records(8)))[0]

    assert y.dtype == np.int32


@requires_mlx
def test_a_multilabel_target_becomes_a_multi_hot_matrix() -> None:
    from recordstream import Stream
    from recordstream.items import Image, MultiLabel

    records = [{"image": Image(np.zeros(INPUT, "float32"), layout="HWC"), "class": MultiLabel([0, 2])}]
    clf = _classifier(num_classes=3)
    clf.multilabel = True

    assert clf._sequence(Stream(source=records))[0][1].tolist() == [[1.0, 0.0, 1.0]]


# --------------------------------------------------------------------------- #
# Name resolution — mlx.nn.losses are functions, mlx.optimizers are CLASSES
# --------------------------------------------------------------------------- #


@requires_mlx
def test_losses_resolve_by_name() -> None:
    import mlx.nn as nn

    from sonair.mlx_classifier import _as_loss

    assert _as_loss("cross_entropy") is nn.losses.cross_entropy
    with pytest.raises(ValueError, match="mlx.nn.losses"):
        _as_loss("nope")


@requires_mlx
def test_multilabel_binds_with_logits() -> None:
    """The head emits raw logits and MLX's `binary_cross_entropy` defaults to probabilities.
    Handed logits it fails the same SILENT way the Keras string losses do: a finite loss that
    simply does not train."""
    from sonair.mlx_classifier import _as_loss

    bound = _as_loss("binary_cross_entropy", multilabel=True)

    assert getattr(bound, "keywords", {}).get("with_logits") is True


@requires_mlx
def test_optimizers_are_CLASSES_here_not_factories() -> None:
    """Unlike optax, MLX ships optimizer classes — so a name is instantiated, not called. The
    shared `learning_rate` knob stays the one place an LR is written either way."""
    import mlx.optimizers as optim

    from sonair.mlx_classifier import _as_optimizer

    made = _as_optimizer("Adam", 0.5)

    assert isinstance(made, optim.Adam)
    # a ready-made optimizer is taken as-is
    assert _as_optimizer(made, 0.1) is made  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="mlx.optimizers"):
        _as_optimizer("nope", 0.1)


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #


@requires_mlx
def test_defaults_are_resolved_from_the_data() -> None:
    from recordstream import Stream

    clf = _classifier(train_set=Stream(source=_records()))
    assert clf.loss is None and clf.metrics is None

    clf.initialize()
    assert clf.loss == "cross_entropy" and clf.metrics == ["accuracy"]


@requires_mlx
def test_fit_trains_and_reports(tmp_path: Path) -> None:
    from recordstream import Stream

    clf = _classifier(train_set=Stream(source=_records(12)), val_set=Stream(source=_records(8)), max_epochs=2)
    clf.metrics_sidecar_path = str(tmp_path / "metrics.json")
    clf.run()

    assert "loss" in clf.computed_metrics and "val_accuracy" in clf.computed_metrics
    assert (tmp_path / "metrics.json").exists()
    assert clf.num_classes == 3


@requires_mlx
def test_the_loss_actually_descends() -> None:
    """The check that catches a loop that runs without learning — which is exactly what a
    missing `mx.eval` or a wrong `with_logits` produces: finite numbers that never improve."""
    from recordstream import Stream

    records = _records(24)
    first = _classifier(train_set=Stream(source=records), max_epochs=1)
    first.run()
    longer = _classifier(train_set=Stream(source=records), max_epochs=6)
    longer.run()

    assert longer.computed_metrics["loss"] < first.computed_metrics["loss"]


@requires_mlx
def test_the_per_epoch_reshuffle_happens() -> None:
    """A `fit()` calls this between epochs; a hand-written loop must. Forgetting it is
    INVISIBLE — training silently runs on one fixed row order."""
    from recordstream import Stream

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

    assert len(seen) == 3


@requires_mlx
def test_string_targets_are_encoded_through_a_fitted_label_map() -> None:
    from recordstream import Stream

    clf = _classifier(train_set=Stream(source=_records(9, labels="names")))
    clf.run()

    assert clf.num_classes == 3 and clf.class_names == ["bird", "cat", "dog"]


# --------------------------------------------------------------------------- #
# Predictions + outputs
# --------------------------------------------------------------------------- #


class _CaptureSink:
    def __init__(self) -> None:
        self.rows: List[Tuple[Any, Any]] = []
        self.closed = False

    def write(self, prediction: Any, metadata: Any) -> None:
        self.rows.append((prediction, metadata))

    def close(self) -> None:
        self.closed = True


@requires_mlx
def test_predict_writes_one_entry_per_record_with_its_metadata() -> None:
    from confluid import LazyClass
    from recordstream import Stream

    from sonair.convnet.mlx_convnet import MlxConvNet

    sink = _CaptureSink()
    clf = _classifier(
        model=LazyClass(MlxConvNet, input_shape=INPUT, num_classes=3),
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


@requires_mlx
def test_discovery_tags_and_outputs() -> None:
    from confluid import output_specs

    from sonair.mlx_classifier import MlxClassifier

    assert MlxClassifier.__confluid_task__ == "classification"
    assert MlxClassifier.__confluid_framework__ == "mlx"
    names = {spec["name"] for spec in output_specs(MlxClassifier)}
    assert {"trained_model", "computed_metrics", "class_names"} <= names


@requires_mlx
@pytest.mark.parametrize("method", ["test", "predict"])
def test_eval_verbs_without_a_test_set_are_clear_errors(method: str) -> None:
    from sonair.convnet.mlx_convnet import MlxConvNet

    clf = _classifier(model=MlxConvNet(num_classes=3, input_shape=INPUT), task=method)
    with pytest.raises(ValueError, match="test_set"):
        getattr(clf, method)()
