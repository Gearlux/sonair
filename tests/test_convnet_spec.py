"""The shared conv spec — one architecture, implemented per framework.

This file is the reason `sonair/convnet/spec.py` exists. Five hand-written implementations of
"the same" network agree only by accident unless something checks, and the thing worth checking
is not that each runs but that they are the SAME network. The parameter count is derived from
the spec rather than measured from one implementation, so no framework is the reference and a
drift in any of them fails here.

Each implementation is skipped when its framework is absent — they are selectable backends, not
core dependencies — so this file reports on whatever is installed rather than requiring all five.
"""

from typing import Any, Callable, Dict, List, Tuple

import pytest

from sonair.convnet.spec import CONV_WIDTHS, INPUT_SHAPE, KERNEL_SIZE, POOL_SIZE, trainable_parameter_count

NUM_CLASSES = 10
INPUT = (16, 16, 3)


# --------------------------------------------------------------------------- #
# The spec's own arithmetic
# --------------------------------------------------------------------------- #


def test_the_count_is_derived_from_the_architecture() -> None:
    """Spelled out here so the formula is checked against a hand computation, not itself.

    Per block: conv weights (k*k*c_in*w) + conv bias (w) + BatchNorm scale and bias (2w).
    Then the dense head: 128*C + C.
    """
    expected = (
        (3 * 3 * 3 * 32 + 32 + 64)  # block 1: conv + bias + BN
        + (3 * 3 * 32 * 64 + 64 + 128)  # block 2
        + (3 * 3 * 64 * 128 + 128 + 256)  # block 3
        + (128 * NUM_CLASSES + NUM_CLASSES)  # head
    )

    assert trainable_parameter_count(NUM_CLASSES) == expected
    assert expected == 93_696 + 129 * NUM_CLASSES  # the closed form the module docstring quotes


def test_the_constants_are_the_single_source_of_truth() -> None:
    """Each implementation imports these for its defaults, so "same architecture" is true by
    construction for the knobs rather than by five copies agreeing."""
    assert CONV_WIDTHS == (32, 64, 128)
    assert KERNEL_SIZE == 3 and POOL_SIZE == 2
    assert INPUT_SHAPE == (224, 224, 3), "the KNOB is (H, W, C) in all five, whatever the tensor layout"


# --------------------------------------------------------------------------- #
# The implementations, each against the spec
# --------------------------------------------------------------------------- #


def _torch_params() -> int:
    from sonair.convnet.torch_convnet import TorchConvNet

    model = TorchConvNet(num_classes=NUM_CLASSES, input_shape=INPUT)
    model.solidify()
    return int(model.parameter_count)


def _keras_params() -> int:
    import math

    from sonair.convnet.keras_convnet import KerasConvNet

    model = KerasConvNet(num_classes=NUM_CLASSES, input_shape=INPUT)
    # `.shape`, not `.size`: a Keras 3 `Variable` has no `.size` (measured) — it is a backend
    # -neutral handle, not an ndarray.
    return int(sum(math.prod(w.shape) for w in model.model.trainable_weights))


def _flax_params() -> int:
    import jax
    from flax import nnx

    from sonair.convnet.flax_convnet import FlaxConvNet

    model = FlaxConvNet(num_classes=NUM_CLASSES, input_shape=INPUT)
    model.solidify()
    return int(sum(int(v.size) for v in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))))


def _mlx_params() -> int:
    from sonair.convnet.mlx_convnet import MlxConvNet  # type: ignore[import-not-found]  # lands with the mlx backend

    model = MlxConvNet(num_classes=NUM_CLASSES, input_shape=INPUT)
    model.solidify()
    return int(model.parameter_count)


def _tensorflow_params() -> int:
    from sonair.convnet.tensorflow_convnet import (
        TensorFlowConvNet,  # type: ignore[import-not-found]  # lands with the TF models
    )

    model = TensorFlowConvNet(num_classes=NUM_CLASSES, input_shape=INPUT)
    model.solidify()
    return int(model.parameter_count)


#: implementation name -> (the module that must import, how to count its trainable parameters)
IMPLEMENTATIONS: Dict[str, Tuple[str, Callable[[], int]]] = {
    "torch": ("torch", _torch_params),
    "keras": ("keras", _keras_params),
    "flax": ("jax", _flax_params),
    "mlx": ("mlx", _mlx_params),
    "tensorflow": ("tensorflow", _tensorflow_params),
}


def _usable(name: str, module: str) -> bool:
    """Is this implementation both installed AND usable in THIS process?

    Installed is not sufficient for `tensorflow`: Keras binds ONE engine per process, and
    `sonair.convnet.tensorflow_convnet` imports through the shim that pins `KERAS_BACKEND=tensorflow`.
    In a full `pytest` run another test has already bound torch, so building it there fails —
    the same split `tests/test_tensorflow_convnet.py` documents. Alone it passes; that is why the
    condition is the ENGINE, not the install.
    """
    import importlib.util

    if importlib.util.find_spec(module) is None:
        return False
    if name != "tensorflow":
        return True
    try:
        from recordstream.keras import keras_backend
    except ImportError:  # pragma: no cover - keras absent entirely
        return False
    return bool(keras_backend() == "tensorflow")


@pytest.mark.parametrize("name", sorted(IMPLEMENTATIONS))
def test_the_implementation_matches_the_spec(name: str) -> None:
    """Each is checked against the DERIVED count, so none of them is the reference."""
    module, count = IMPLEMENTATIONS[name]
    if not _usable(name, module):
        pytest.skip(f"{name} is not usable in this process (needs sonair[{module}], and the matching keras engine)")

    assert count() == trainable_parameter_count(NUM_CLASSES)


def test_at_least_two_implementations_agree() -> None:
    """Guard against the parametrised test passing vacuously.

    If every implementation skipped, the suite above would be green while checking nothing —
    and "same module across frameworks" is a claim about AGREEMENT, so it needs two.
    """
    counts: List[int] = []
    for name, (module, count) in sorted(IMPLEMENTATIONS.items()):
        if _usable(name, module):
            counts.append(count())

    assert len(counts) >= 2, f"only {len(counts)} implementation(s) installed — install another backend"
    assert len(set(counts)) == 1, f"implementations disagree: {counts}"


def test_every_implementation_takes_the_same_constructor_knobs() -> None:
    """One config block drives any of them by changing the class name alone — which only holds
    if the knobs line up. Checked on whichever are installed."""
    import inspect

    classes: List[Any] = []
    if _usable("torch", "torch"):
        from sonair.convnet.torch_convnet import TorchConvNet

        classes.append(TorchConvNet)
    if _usable("keras", "keras"):
        from sonair.convnet.keras_convnet import KerasConvNet

        classes.append(KerasConvNet)
    if _usable("flax", "jax"):
        from sonair.convnet.flax_convnet import FlaxConvNet

        classes.append(FlaxConvNet)
    if _usable("mlx", "mlx"):
        from sonair.convnet.mlx_convnet import MlxConvNet  # type: ignore[import-not-found]

        classes.append(MlxConvNet)
    if _usable("tensorflow", "tensorflow"):
        from sonair.convnet.tensorflow_convnet import TensorFlowConvNet  # type: ignore[import-not-found]

        classes.append(TensorFlowConvNet)

    assert len(classes) >= 2, "need two installed implementations to compare"
    # `seed` joined the shared set on 2026-08-08, when KerasConvNet — the ONE implementation
    # without it — gained the knob: its sweep rows were the only ones moving between identical
    # runs (the raidar KerasFcos story replayed, and the same fix).
    shared = {"num_classes", "input_shape", "widths", "dropout", "seed"}
    for cls in classes:
        params = set(inspect.signature(cls.__init__).parameters) - {"self"}
        assert shared <= params, f"{cls.__name__} is missing {shared - params}"


def test_the_keras_seed_actually_reproduces_the_weights() -> None:
    """The knob is only worth having if it BINDS — assert on the built weights, not the API."""
    if not _usable("keras", "keras"):
        pytest.skip("keras is not installed")
    import numpy as np

    from sonair.convnet.keras_convnet import KerasConvNet

    def first_kernel(seed: int) -> Any:
        model = KerasConvNet(num_classes=NUM_CLASSES, input_shape=INPUT, seed=seed)
        return np.asarray(model.solidify().get_weights()[0])

    a, b, other = first_kernel(0), first_kernel(0), first_kernel(1)
    assert np.array_equal(a, b), "the same seed must rebuild the same weights"
    assert not np.array_equal(a, other), "a different seed must not — else it is seeding nothing"
