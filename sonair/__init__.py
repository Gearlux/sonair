"""Sonair: image classification across training backends, driven by one config.

**The core is backend-free.** ``pip install sonair`` gives you the CLI and the config surface;
a training framework arrives with the backend you select::

    pip install 'sonair[lightning]'          # PyTorch Lightning — the reference backend
    pip install 'sonair[torch]'              # a hand-written loop, no training framework
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

from matrainer._exports import lazy_exports

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
#
# The two kinds of model wrap different things — somebody's zoo, and nothing at all — and only
# the second is the cross-framework comparison, so they live apart: a zoo wrapper in
# ``<backend>_model.py``, and every implementation of the shared conv spec in the
# :mod:`sonair.convnet` package (which imports nothing itself, so these stay lazy).
_BACKEND_NAMES = {
    "LightningClassifier": ("sonair.lightning_classifier", "lightning"),
    "TorchClassifier": ("sonair.torch_classifier", "torch"),
    "TimmModel": ("sonair.timm_model", "lightning"),
    "KerasClassifier": ("sonair.keras_classifier", "keras"),
    "KerasModel": ("sonair.keras_model", "keras"),
    "TensorFlowClassifier": ("sonair.tensorflow_classifier", "tensorflow"),
    "TensorFlowModel": ("sonair.tensorflow_model", "tensorflow"),
    "FastaiClassifier": ("sonair.fastai_classifier", "fastai"),
    "HuggingFaceClassifier": ("sonair.huggingface_classifier", "huggingface"),
    "HuggingFaceModel": ("sonair.huggingface_model", "huggingface"),
    "JaxClassifier": ("sonair.jax_classifier", "jax"),
    "MlxClassifier": ("sonair.mlx_classifier", "mlx"),
    # The five implementations of `sonair/convnet/spec.py` — the same network in five idioms,
    # each an instance of its framework's own model type. `TorchConvNet` serves EVERY torch
    # backend (lightning, torch, fastai — they differ in the loop, not in what a model is),
    # which is why there is no separate name per backend; the huggingface backend has none at
    # all, since `transformers.Trainer` drives models that compute their own loss (see the
    # convnet-package mandate in AGENTS.md). Its extra is the SMALLEST one that supplies what
    # it needs — plain `torch`, not `lightning`, so the ImportError advises the smaller install.
    "TorchConvNet": ("sonair.convnet.torch_convnet", "torch"),
    "KerasConvNet": ("sonair.convnet.keras_convnet", "keras"),
    "TensorFlowConvNet": ("sonair.convnet.tensorflow_convnet", "tensorflow"),
    "FlaxConvNet": ("sonair.convnet.flax_convnet", "jax"),
    "MlxConvNet": ("sonair.convnet.mlx_convnet", "mlx"),
    # The one loss/metric class sonair OWNS — every other entry in `keras_classification` is an
    # upstream class registered as-is and reached by its own name. It is here because a config
    # writes `!class:sonair.SparseF1Score`, and it must be lazy for the same reason the rest
    # are: naming it must not import keras on a torch-only install.
    "SparseF1Score": ("sonair.keras_classification", "keras"),
}

__all__ = [*_BACKEND_NAMES]

# The resolver is matrainer's `lazy_exports` (the machinery was identical in every training
# project's root); the `_BACKEND_NAMES` table above is the project DATA it runs on.
__getattr__, __dir__ = lazy_exports(__name__, _BACKEND_NAMES)
