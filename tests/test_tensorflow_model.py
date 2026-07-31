# mypy: disable-error-code="attr-defined"
"""The TensorFlow-native models: the SavedModel zoo wrapper and the raw-TF conv spec.

The claim worth pinning here is narrow and specific. `docs/architecture.md` §5 said the
TensorFlow backend contributes no model module because a `TensorFlowModel` would be
`KerasModel` renamed — true of a zoo wrapper over `keras.applications`, and false of these two.
So these tests assert what makes them NOT duplicates: a raw-`tf.nn` implementation of the
shared spec, and a loader for a genuinely different artifact format.

Needs a Keras bound to the TensorFlow engine. A full `pytest` run binds torch first
(alphabetically), so most of this file skips there and CI runs it as its own invocation:

    KERAS_BACKEND=tensorflow pytest tests/test_tensorflow_model.py
"""

import importlib.util
from pathlib import Path
from typing import Any

import pytest

from sonair.convnet_spec import trainable_parameter_count

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
    reason="needs keras on the TensorFlow engine: KERAS_BACKEND=tensorflow pytest tests/test_tensorflow_model.py",
)


# --------------------------------------------------------------------------- #
# Source-only — these must keep running where TensorFlow is ABSENT
# --------------------------------------------------------------------------- #
# Same reasoning as the TF classifier suite: the machines that can break the import ordering or
# drop a class are exactly the ones without TensorFlow installed.


def test_the_module_imports_tf_through_the_pinning_shim() -> None:
    """`sonair._tensorflow` pins KERAS_BACKEND=tensorflow BEFORE keras loads. Importing `tf`
    any other way here would let an import sorter reorder the pin out of effect."""
    source = (Path(__file__).resolve().parents[1] / "sonair" / "tensorflow_model.py").read_text()

    assert "from sonair._tensorflow import tf" in source
    assert "\nimport tensorflow" not in source, "must not import tensorflow directly"


def test_both_classes_are_exported() -> None:
    source = (Path(__file__).resolve().parents[1] / "sonair" / "tensorflow_model.py").read_text()

    assert '__all__ = ["TensorFlowConvNet", "TensorFlowModel"]' in source


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
    from sonair.tensorflow_model import TensorFlowConvNet

    model = TensorFlowConvNet(num_classes=NUM_CLASSES, input_shape=INPUT)
    model.solidify()

    assert model.parameter_count == trainable_parameter_count(NUM_CLASSES)


@requires_tensorflow
@requires_tf_engine
def test_batchnorm_statistics_are_state_not_parameters() -> None:
    """Moving mean/variance are `trainable=False`, which is what lets the count line up: the
    four frameworks disagree about whether to include them, so the spec counts neither."""
    from sonair.tensorflow_model import TensorFlowConvNet

    model = TensorFlowConvNet(num_classes=NUM_CLASSES, input_shape=INPUT)
    model.solidify()
    inner = model.model

    assert len(inner.variables) > len(inner.trainable_variables), "the moving stats must exist"
    assert len(inner.variables) - len(inner.trainable_variables) == 2 * 3, "two per conv block"


@requires_tensorflow
@requires_tf_engine
def test_zero_arg_construction_allocates_nothing() -> None:
    from sonair.tensorflow_model import TensorFlowConvNet

    model = TensorFlowConvNet()

    assert model._model is None
    with pytest.raises(ValueError, match="num_classes"):
        model.solidify()


@requires_tensorflow
@requires_tf_engine
def test_the_head_emits_raw_logits() -> None:
    """No activation, like every other head — so `from_logits` losses apply unchanged."""
    import numpy as np

    from sonair.tensorflow_model import TensorFlowConvNet

    model = TensorFlowConvNet(num_classes=NUM_CLASSES, input_shape=INPUT)
    out = np.asarray(model(np.zeros((2, *INPUT), "float32")))

    assert not np.allclose(out.sum(axis=-1), 1.0)


@requires_tensorflow
@requires_tf_engine
def test_training_updates_the_moving_statistics() -> None:
    """Written by hand rather than borrowed from `keras.layers`, so the update is ours to get
    right — and an un-updated moving mean makes eval silently disagree with training."""
    import numpy as np

    from sonair.tensorflow_model import TensorFlowConvNet

    model = TensorFlowConvNet(num_classes=NUM_CLASSES, input_shape=INPUT)
    block = model.model.blocks[0]
    before = np.asarray(block.moving_mean).copy()

    model(np.random.default_rng(0).random((4, *INPUT)).astype("float32"), training=True)
    after = np.asarray(block.moving_mean)

    assert not np.allclose(before, after), "training must fold batch statistics into the estimate"


# --------------------------------------------------------------------------- #
# TensorFlowModel — the SavedModel zoo
# --------------------------------------------------------------------------- #


@requires_tensorflow
@requires_tf_engine
def test_a_missing_or_unset_path_is_a_clear_error() -> None:
    from sonair.tensorflow_model import TensorFlowModel

    with pytest.raises(ValueError, match="saved_model is not set"):
        TensorFlowModel().solidify()
    with pytest.raises(ValueError, match="does not exist"):
        TensorFlowModel(saved_model="/nonexistent/path").solidify()


@pytest.fixture
def exported_convnet(tmp_path: Path) -> Any:
    """A SavedModel this project produced, via the classifier's own export path."""
    import numpy as np

    from sonair._tensorflow import tf
    from sonair.tensorflow_model import TensorFlowConvNet

    model = TensorFlowConvNet(num_classes=NUM_CLASSES, input_shape=INPUT)
    model.solidify()
    path = tmp_path / "saved"
    spec = tf.TensorSpec(shape=(None, *INPUT), dtype=tf.float32)
    call = tf.function(lambda x: model(x, training=False)).get_concrete_function(spec)
    tf.saved_model.save(model, str(path), signatures={"serving_default": call})
    assert (path / "saved_model.pb").exists()
    _ = np
    return path


@requires_tensorflow
@requires_tf_engine
def test_a_tf_module_export_round_trips(exported_convnet: Path) -> None:
    """The round trip that makes this a real zoo rather than a hypothesis: export from this
    project, load it back, and get the same parameter count and a working forward pass."""
    import numpy as np

    from sonair.tensorflow_model import TensorFlowModel

    loaded = TensorFlowModel(saved_model=str(exported_convnet), num_classes=NUM_CLASSES)

    assert loaded.parameter_count == trainable_parameter_count(NUM_CLASSES)
    out = np.asarray(loaded(np.zeros((2, *INPUT), "float32")))
    assert out.shape == (2, NUM_CLASSES)


@requires_tensorflow
@requires_tf_engine
def test_variables_are_found_even_though_load_drops_the_property(exported_convnet: Path) -> None:
    """The subtle one. `tf.saved_model.load` restores tracked DATA, not the Python class, so
    `tf.Module.variables` — a computed property — is simply GONE from the restored object. A
    naive `getattr(loaded, "trainable_variables")` finds nothing and the model looks
    un-fine-tunable. Measured: the module export reports none while its children hold all of
    them; a Keras export reports its own list.
    """
    from sonair._tensorflow import tf
    from sonair.tensorflow_model import _restored_trainable_variables

    raw = tf.saved_model.load(str(exported_convnet))

    assert not hasattr(raw, "trainable_variables"), "if this ever gains one, simplify the walk"
    assert len(_restored_trainable_variables(raw)) == 14, "3 blocks x 4 + head kernel + bias"


@requires_tensorflow
@requires_tf_engine
def test_an_inference_only_export_refuses_to_be_fine_tuned(tmp_path: Path) -> None:
    """A silent failure otherwise: training would apply empty gradients forever and the loss
    would simply never move. Caught at BUILD time, with the flag that makes it loadable."""
    from sonair._tensorflow import tf
    from sonair.tensorflow_model import TensorFlowModel

    class _NoVariables(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(None, 2), dtype=tf.float32)])
        def __call__(self, x: Any) -> Any:
            return x * 2.0

    path = tmp_path / "frozen"
    tf.saved_model.save(_NoVariables(), str(path))

    with pytest.raises(ValueError, match="cannot be fine-tuned"):
        TensorFlowModel(saved_model=str(path), trainable=True).solidify()

    # ...and it loads fine for inference, which is the documented way out.
    assert TensorFlowModel(saved_model=str(path), trainable=False).solidify() is not None


@requires_tensorflow
@requires_tf_engine
def test_discovery_tags_put_them_in_the_tensorflow_backends_picker() -> None:
    """`framework="keras"`, not `"tensorflow"`: the axis names the API a class is WIRED TO, and
    `TensorFlowClassifier` — the only trainer that consumes these — declares `"keras"` itself.
    Tagging them `"tensorflow"` would hide them from the one picker that should offer them."""
    from sonair.tensorflow_classifier import TensorFlowClassifier
    from sonair.tensorflow_model import TensorFlowConvNet, TensorFlowModel

    for cls in (TensorFlowConvNet, TensorFlowModel):
        assert cls.__confluid_task__ == "classification"
        assert cls.__confluid_role__ == "model"
        assert cls.__confluid_framework__ == TensorFlowClassifier.__confluid_framework__ == "keras"
