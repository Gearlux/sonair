"""The ONE place Keras is imported — so the backend is chosen before it is.

Keras 3 reads ``KERAS_BACKEND`` **at import time** and defaults to ``tensorflow``. sonair does
not require TensorFlow (its `keras` extra installs Keras alone), so an operator who has not
chosen a backend would hit ``ModuleNotFoundError: No module named 'tensorflow'`` from deep
inside a ``keras.utils`` import — a confusing failure that says nothing about what to do.

So this module sets a default the package can actually honour — ``torch``, which sonair always
has — and only when the operator has not chosen: ``setdefault`` never overrides an explicit
``KERAS_BACKEND=tensorflow`` / ``=jax``. Every sonair module imports keras THROUGH here
(``from sonair._keras import keras``) so the ordering cannot be got wrong by an import that
happens to land first.

Choosing the backend is still the operator's call. What this removes is the failure mode where
NOT choosing means a crash instead of a working default.
"""

import os

#: Set BEFORE keras is imported, and only if unset — an explicit choice always wins.
os.environ.setdefault("KERAS_BACKEND", "torch")

import keras  # noqa: E402 - deliberately after the environment default above

__all__ = ["keras", "keras_backend"]


def keras_backend() -> str:
    """The Keras backend actually in effect (``torch`` / ``tensorflow`` / ``jax``).

    Reads Keras rather than the environment variable, so it reports what is LOADED — Keras
    consults ``KERAS_BACKEND`` once at import, and changing it afterwards has no effect. This
    is the first thing a multi-backend demo should log at run start.
    """
    return str(keras.backend.backend())
