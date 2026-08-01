# mypy: disable-error-code="attr-defined"
"""`TensorFlowConvNet` — the shared conv spec in raw `tf.nn` ops.

The claim worth pinning here is narrow and specific. `docs/architecture.md` §5 said the
TensorFlow backend contributes no model module because a `TensorFlowModel` would be
`KerasModel` renamed — true of a zoo wrapper over `keras.applications`, and false of this. So
these tests assert what makes it NOT a duplicate: a raw-`tf.nn` implementation of the shared
spec, wrapping nothing at all. The other half of that amendment — the SavedModel loader, which
wraps a genuinely different zoo — is `tests/test_tensorflow_model.py`.

Needs a Keras bound to the TensorFlow engine. A full `pytest` run binds torch first
(alphabetically), so most of this file skips there and CI runs it as its own invocation:

    KERAS_BACKEND=tensorflow pytest tests/test_tensorflow_convnet.py
"""

import importlib.util
from pathlib import Path

import pytest

from sonair.convnet.spec import trainable_parameter_count

INPUT = (16, 16, 3)
NUM_CLASSES = 4

requires_tensorflow = pytest.mark.skipif(
    importlib.util.find_spec("tensorflow") is None,
    reason="the TensorFlow models need sonair[tensorflow]",
)


def _tf_engine() -> bool:
    """Is Keras bound to the TensorFlow engine in THIS process?"""
    try:
        from recordstream.keras import keras_backend
    except ImportError:  # pragma: no cover - keras absent entirely
        return False
    return keras_backend() == "tensorflow"


requires_tf_engine = pytest.mark.skipif(
    not _tf_engine(),
    reason="needs keras on the TensorFlow engine: KERAS_BACKEND=tensorflow pytest tests/test_tensorflow_convnet.py",
)


# --------------------------------------------------------------------------- #
# Source-only — these must keep running where TensorFlow is ABSENT
# --------------------------------------------------------------------------- #
# Same reasoning as the TF classifier suite: the machines that can break the import ordering or
# drop a class are exactly the ones without TensorFlow installed.


def test_the_module_imports_tf_through_the_pinning_shim() -> None:
    """`sonair._tensorflow` pins KERAS_BACKEND=tensorflow BEFORE keras loads. Importing `tf`
    any other way here would let an import sorter reorder the pin out of effect."""
    source = (Path(__file__).resolve().parents[1] / "sonair" / "convnet" / "tensorflow_convnet.py").read_text()

    assert "from sonair._tensorflow import tf" in source
    assert "\nimport tensorflow" not in source, "must not import tensorflow directly"


def test_the_class_is_exported() -> None:
    source = (Path(__file__).resolve().parents[1] / "sonair" / "convnet" / "tensorflow_convnet.py").read_text()

    assert '__all__ = ["TensorFlowConvNet"]' in source


def test_the_classifier_branches_on_the_export_capability() -> None:
    """A `tf.Module` has no `keras.Model.export`, so the classifier must not call it blindly."""
    source = (Path(__file__).resolve().parents[1] / "sonair" / "tensorflow_classifier.py").read_text()

    assert 'hasattr(self.model, "export")' in source
    assert "tf.saved_model.save" in source


# --------------------------------------------------------------------------- #
# TensorFlowConvNet — the shared spec in raw tf.nn
# --------------------------------------------------------------------------- #


@requires_tensorflow
@requires_tf_engine
def test_it_matches_the_shared_spec() -> None:
    """The point of implementing the architecture a fourth time: the count is DERIVED from the
    spec, so this is checked against the claim rather than against another framework."""
    from sonair.convnet.tensorflow_convnet import TensorFlowConvNet

    model = TensorFlowConvNet(num_classes=NUM_CLASSES, input_shape=INPUT)
    model.solidify()

    assert model.parameter_count == trainable_parameter_count(NUM_CLASSES)


@requires_tensorflow
@requires_tf_engine
def test_batchnorm_statistics_are_state_not_parameters() -> None:
    """Moving mean/variance are `trainable=False`, which is what lets the count line up: the
    four frameworks disagree about whether to include them, so the spec counts neither."""
    from sonair.convnet.tensorflow_convnet import TensorFlowConvNet

    model = TensorFlowConvNet(num_classes=NUM_CLASSES, input_shape=INPUT)
    model.solidify()
    inner = model.model

    assert len(inner.variables) > len(inner.trainable_variables), "the moving stats must exist"
    assert len(inner.variables) - len(inner.trainable_variables) == 2 * 3, "two per conv block"


@requires_tensorflow
@requires_tf_engine
def test_zero_arg_construction_allocates_nothing() -> None:
    from sonair.convnet.tensorflow_convnet import TensorFlowConvNet

    model = TensorFlowConvNet()

    assert model._model is None
    with pytest.raises(ValueError, match="num_classes"):
        model.solidify()


@requires_tensorflow
@requires_tf_engine
def test_the_head_emits_raw_logits() -> None:
    """No activation, like every other head — so `from_logits` losses apply unchanged."""
    import numpy as np

    from sonair.convnet.tensorflow_convnet import TensorFlowConvNet

    model = TensorFlowConvNet(num_classes=NUM_CLASSES, input_shape=INPUT)
    out = np.asarray(model(np.zeros((2, *INPUT), "float32")))

    assert not np.allclose(out.sum(axis=-1), 1.0)


@requires_tensorflow
@requires_tf_engine
def test_training_updates_the_moving_statistics() -> None:
    """Written by hand rather than borrowed from `keras.layers`, so the update is ours to get
    right — and an un-updated moving mean makes eval silently disagree with training."""
    import numpy as np

    from sonair.convnet.tensorflow_convnet import TensorFlowConvNet

    model = TensorFlowConvNet(num_classes=NUM_CLASSES, input_shape=INPUT)
    block = model.model.blocks[0]
    before = np.asarray(block.moving_mean).copy()

    model(np.random.default_rng(0).random((4, *INPUT)).astype("float32"), training=True)
    after = np.asarray(block.moving_mean)

    assert not np.allclose(before, after), "training must fold batch statistics into the estimate"


@requires_tensorflow
@requires_tf_engine
def test_discovery_tags_put_it_in_the_tensorflow_backends_picker() -> None:
    """`framework="keras"`, not `"tensorflow"`: the axis names the API a class is WIRED TO, and
    `TensorFlowClassifier` — the only trainer that consumes this — declares `"keras"` itself.
    Tagging it `"tensorflow"` would hide it from the one picker that should offer it."""
    from sonair.convnet.tensorflow_convnet import TensorFlowConvNet
    from sonair.tensorflow_classifier import TensorFlowClassifier

    assert TensorFlowConvNet.__confluid_task__ == "classification"
    assert TensorFlowConvNet.__confluid_role__ == "model"
    assert TensorFlowConvNet.__confluid_framework__ == TensorFlowClassifier.__confluid_framework__ == "keras"
