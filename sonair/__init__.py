"""Sonair: image classification across training backends, driven by one config.

**The core is backend-free.** ``pip install sonair`` gives you the CLI and the config surface;
a training framework arrives with the backend you select::

    pip install 'sonair[lightning]'          # PyTorch Lightning — the reference backend
    pip install 'sonair[keras]'              # Keras 3 (runs on torch / tensorflow / jax)
    pip install 'sonair[tensorflow]'         # a native tf.data + GradientTape loop
    pip install 'sonair[fastai]'             # a fastai Learner over the same torch stack
    pip install 'sonair[huggingface]'        # transformers Trainer, fine-tuning a Hub model
    pip install 'sonair[jax]'                # a jitted optax loop over a Flax model
    pip install 'sonair[mlx]'                # Apple-silicon unified memory (macOS arm64)
    pip install 'sonair[lightning,keras]'    # any combination

Which is why every backend name below resolves LAZILY (PEP 562): importing ``sonair`` must not
require a framework the operator did not choose, so a Keras-only install never drags in torch —
nor the reverse. Touching a name whose backend is missing raises an :class:`ImportError` naming
the exact extra, instead of a traceback from three libraries down.
"""

import importlib
from typing import Any

__version__ = "0.2.0"

#: Backend-owned name -> (defining module, the extra that installs it). Resolved on first
#: attribute access. Keep in step with the extras in ``pyproject.toml``: a name here whose
#: extra does not exist would advertise an install command that fails.
#
# fastai deliberately contributes NO model wrapper: it is torch, so ``FastaiClassifier`` takes
# ``TimmModel``. A backend that adds no new wiring surface adds no new name here
# (``docs/architecture.md`` §5).
#
# TensorFlow used to be in that sentence too, and the reasoning was narrower than it looked: a
# zoo wrapper over ``tf.keras.applications`` WOULD be ``KerasModel`` renamed, but a SavedModel
# loader and a raw-``tf.nn`` net duplicate nothing. Hence the two names below.
_BACKEND_NAMES = {
    "LightningClassifier": ("sonair.lightning_classifier", "lightning"),
    "TimmModel": ("sonair.timm_model", "lightning"),
    "KerasClassifier": ("sonair.keras_classifier", "keras"),
    "KerasModel": ("sonair.keras_model", "keras"),
    "TensorFlowClassifier": ("sonair.tensorflow_classifier", "tensorflow"),
    "FastaiClassifier": ("sonair.fastai_classifier", "fastai"),
    "HuggingFaceClassifier": ("sonair.huggingface_classifier", "huggingface"),
    "HuggingFaceModel": ("sonair.huggingface_model", "huggingface"),
    "JaxClassifier": ("sonair.jax_classifier", "jax"),
    "FlaxModel": ("sonair.flax_model", "jax"),
    "KerasConvNet": ("sonair.keras_model", "keras"),
    "TensorFlowModel": ("sonair.tensorflow_model", "tensorflow"),
    "TensorFlowConvNet": ("sonair.tensorflow_model", "tensorflow"),
    "MlxClassifier": ("sonair.mlx_classifier", "mlx"),
    "MlxModel": ("sonair.mlx_model", "mlx"),
}

__all__ = [*_BACKEND_NAMES]


def __getattr__(name: str) -> Any:
    """Resolve a backend-owned name on first use, or explain which extra is missing."""
    entry = _BACKEND_NAMES.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, extra = entry
    try:
        return getattr(importlib.import_module(module_path), name)
    except ImportError as exc:
        # The failure is a MISSING BACKEND, not a broken sonair — say so, and say what to run.
        # Chained (`from exc`) so the underlying "No module named 'timm'" stays visible.
        raise ImportError(
            f"sonair.{name} needs the {extra!r} backend, which is not installed.\n"
            f"  pip install 'sonair[{extra}]'\n"
            f"(underlying import error: {exc})"
        ) from exc


def __dir__() -> Any:
    """Advertise the lazy names to ``dir()`` / tab-completion."""
    return sorted({*globals(), *_BACKEND_NAMES})
