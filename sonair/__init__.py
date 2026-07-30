"""Sonair: image classification across training backends, driven by one config.

**The core is backend-free.** ``pip install sonair`` gives you the CLI and the config surface;
a training framework arrives with the backend you select::

    pip install 'sonair[lightning]'          # PyTorch Lightning — the reference backend
    pip install 'sonair[keras]'              # Keras 3 (runs on torch / tensorflow / jax)
    pip install 'sonair[lightning,keras]'    # both

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
_BACKEND_NAMES = {
    "LightningClassifier": ("sonair.lightning_classifier", "lightning"),
    "TimmModel": ("sonair.timm_model", "lightning"),
    "KerasClassifier": ("sonair.keras_classifier", "keras"),
    "KerasModel": ("sonair.keras_model", "keras"),
    "RecordSequence": ("sonair.keras_classifier", "keras"),
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
