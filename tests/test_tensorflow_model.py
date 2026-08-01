# mypy: disable-error-code="attr-defined"
"""`TensorFlowModel` — the SavedModel zoo wrapper.

The claim worth pinning here is narrow and specific. `docs/architecture.md` §5 said the
TensorFlow backend contributes no model module because a `TensorFlowModel` would be
`KerasModel` renamed — true of a zoo wrapper over `keras.applications`, and false of this one:
it loads a genuinely different artifact format. The other half of that amendment — the raw-
`tf.nn` implementation of the shared conv spec — is `tests/test_tensorflow_convnet.py`, and
this file borrows that class only to produce a SavedModel to load back.

Needs a Keras bound to the TensorFlow engine. A full `pytest` run binds torch first
(alphabetically), so most of this file skips there and CI runs it as its own invocation:

    KERAS_BACKEND=tensorflow pytest tests/test_tensorflow_model.py
"""

import importlib.util
from pathlib import Path
from typing import Any

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


def test_the_class_is_exported() -> None:
    source = (Path(__file__).resolve().parents[1] / "sonair" / "tensorflow_model.py").read_text()

    assert '__all__ = ["TensorFlowModel"]' in source


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
    from sonair.convnet.tensorflow_convnet import TensorFlowConvNet

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
def test_discovery_tags_put_it_in_the_tensorflow_backends_picker() -> None:
    """`framework="keras"`, not `"tensorflow"`: the axis names the API a class is WIRED TO, and
    `TensorFlowClassifier` — the only trainer that consumes this — declares `"keras"` itself.
    Tagging it `"tensorflow"` would hide it from the one picker that should offer it."""
    from sonair.tensorflow_classifier import TensorFlowClassifier
    from sonair.tensorflow_model import TensorFlowModel

    assert TensorFlowModel.__confluid_task__ == "classification"
    assert TensorFlowModel.__confluid_role__ == "model"
    assert TensorFlowModel.__confluid_framework__ == TensorFlowClassifier.__confluid_framework__ == "keras"
