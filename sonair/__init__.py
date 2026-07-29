"""Sonair: image classification on two engines, driven by one config.

``LightningClassifier`` (PyTorch Lightning) is the reference backend and imports eagerly —
torch is a hard dependency. ``KerasClassifier`` / ``KerasModel`` sit behind the optional
``sonair[keras]`` extra and resolve LAZILY via PEP 562, so ``import sonair`` never requires
keras and a torch-only install is unaffected. Touching either name imports its backend; a
missing one raises the usual ``ImportError`` naming the extra to install.
"""

from typing import Any

from sonair.lightning_classifier import LightningClassifier
from sonair.timm_model import TimmModel

__version__ = "0.2.0"

#: Optional-backend names -> the module that defines them. Resolved on first attribute access.
_LAZY = {
    "KerasClassifier": "sonair.keras_classifier",
    "KerasModel": "sonair.keras_model",
    "RecordSequence": "sonair.keras_classifier",
}

__all__ = [
    "TimmModel",
    "LightningClassifier",
    "KerasClassifier",
    "KerasModel",
]


def __getattr__(name: str) -> Any:
    """Resolve an optional-backend name on first use (PEP 562)."""
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_path), name)


def __dir__() -> Any:
    """Advertise the lazy names to ``dir()`` / tab-completion."""
    return sorted({*globals(), *_LAZY})
