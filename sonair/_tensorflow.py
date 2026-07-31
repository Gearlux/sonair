"""The ONE place sonair imports TensorFlow — and the ``KERAS_BACKEND`` ordering that must precede it.

The twin of :mod:`sonair._keras`, and it exists for the mirror image of that module's problem.
:mod:`recordstream.keras` defaults ``KERAS_BACKEND`` to the first backend actually INSTALLED, in
the order ``torch`` -> ``tensorflow`` -> ``jax``. That is the right answer for a Keras run — Keras
3 is an API and any engine underneath will do — but it is the WRONG one here: on a machine
carrying both ``sonair[lightning]`` and ``sonair[tensorflow]`` (the workspace's own setup), torch
wins, and :class:`~sonair.tensorflow_classifier.TensorFlowClassifier` would then drive a
torch-backed ``keras.Model`` with ``tf.GradientTape``, which cannot work.

Keras reads the variable ONCE, at import time, so the default has to be in place before ANY
module imports keras. Which is why this module hands over the ``keras`` handle **as well as**
``tf``: a consumer that imported both from their "natural" homes would write
``from sonair._keras import keras`` beside ``from sonair._tensorflow import tf``, and isort
sorts ``_keras`` ahead of ``_tensorflow`` — importing keras one line before the default that
was supposed to steer it. Re-exporting here removes the ordering hazard instead of documenting
it: :mod:`sonair.tensorflow_classifier` imports keras, TensorFlow and the backend probe from
this one module, and ``tests/test_tensorflow_classifier.py`` pins that it stays that way.

``setdefault``, not an assignment: an explicit ``KERAS_BACKEND=jax`` still wins, and
``TensorFlowClassifier.initialize()`` then fails with a message naming the mismatch instead of
running on an engine its training loop cannot drive.
"""

import os

# BEFORE the imports below, and before anything else imports keras — see the module docstring
# for why this cannot live at the bottom of an import block.
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import tensorflow as tf  # noqa: E402 - deliberately after the environment default above

# Through recordstream, exactly as `sonair/_keras.py` does — same one spelling for the handle,
# now that the variable it reads is already pinned.
from recordstream.keras import keras, keras_backend  # noqa: E402

__all__ = ["keras", "keras_backend", "tf"]
