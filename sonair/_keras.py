"""The ONE place sonair imports Keras — which is now recordstream's boundary, not sonair's.

Keras 3 reads ``KERAS_BACKEND`` **at import time** and defaults to ``tensorflow``, which
``sonair[keras]`` does not install, so a bare ``import keras`` dies with
``ModuleNotFoundError: No module named 'tensorflow'`` from inside ``keras.src.tree``. The fix is
to set a default the install can honour BEFORE keras is imported — and that has to happen in the
LOWEST layer that imports keras, because import sorters group a library import ahead of a
first-party one: with the batching adapter living in ``recordstream.keras``, a
``from recordstream.keras import RecordSequence`` line sorts ABOVE
``from sonair._keras import keras`` and would have imported keras first.

So :mod:`recordstream.keras` owns the ordering (and the ``_first_installed_backend`` probe that
picks it), and this module re-exports its handles. A sonair module reaches keras through either
``sonair._keras`` or ``recordstream.keras`` — the same objects, the same ordering — and never
``import keras`` directly.
"""

from recordstream.keras import keras, keras_backend

__all__ = ["keras", "keras_backend"]
