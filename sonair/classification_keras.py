"""``ClassificationKerasHooks`` — the classification task's KERAS-family hook implementations.

The keras twin of :mod:`sonair.classification_torch`: the task hooks written once for BOTH
Keras-family machinery bases — the compile/fit member (:class:`matrainer.keras.KerasRunnable`)
and the native-TensorFlow loop (:class:`matrainer.tensorflow.TensorFlowRunnable`) — so the two
concretes cannot drift apart on what classification means in the keras vocabulary. A concrete
backend composes ``(ClassificationKerasHooks, <Machinery>Runnable)``; the label pipeline
itself (the multilabel peek, ``LabelMap`` fitting/encoding, ``num_classes`` resolution) is
:class:`~sonair.classification_base.ClassificationBase`'s, which this class extends.

What the task says, in this vocabulary:

* **The default loss branches on the peeked target shape** — ``BinaryCrossentropy`` for a
  multi-label run, ``SparseCategoricalCrossentropy`` otherwise, BOTH ``from_logits=True``: the
  head is a bare ``Dense`` emitting raw logits, and Keras's string losses default to
  probabilities — handed logits they normalize by the row SUM and CLIP rather than raising, so
  the loss stays finite and merely wrong (a real MNIST run diverged to ``nan`` before this was
  found).
* **The default metric is the accuracy OBJECT named ``accuracy``** — ``compile()`` would
  expand the ``"accuracy"`` alias contextually and name the column after whatever it picked
  (``sparse_categorical_accuracy``), a compile-time artefact no other engine can produce. The
  multi-label branch takes ``threshold=0.0`` — the logit-space equivalent of the probability
  default 0.5 (this UNIFIES the two concretes: the compile/fit one used to omit it, silently
  thresholding logits at 0.5).
* **The batch target is ``multi_hot`` or int64 ids** — the same split the torch family spells
  as float multi-hot vs ``Long``.
* **A prediction is the :class:`~recordstream.ClassificationOutput` contract** — logits,
  row-softmax probs, argmax ids; numpy, so the SAME sink that consumes the torch backends'
  output consumes it with no adapter.
* **The class names persist BEFORE the fit** (:meth:`_before_fit`) — this family hands the run
  lifecycle to the tracking callback, whose ``on_train_end`` closes the run out, so a
  tracker-touching artifact write after ``fit()`` returns would arrive at an ENDED run.
"""

from pathlib import Path
from typing import Any, List, Optional, cast

import numpy as np
from confluid import Lazy
from loggair import get_logger
from matrainer.artifacts import save_class_names
from matrainer.keras.runnable import KerasLoss, KerasMetric
from recordstream import ClassificationOutput, batch_values, class_names, multi_hot
from recordstream.keras import keras

from sonair.classification_base import ClassificationBase

logger = get_logger(__name__)

__all__ = ["ClassificationKerasHooks"]


class ClassificationKerasHooks(ClassificationBase):
    """The keras-typed classification hooks, shared by the Keras and TensorFlow concretes."""

    def _default_keras_loss(self) -> Optional[Lazy[KerasLoss]]:
        return (
            keras.losses.BinaryCrossentropy(from_logits=True)
            if self.multilabel
            else keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        )

    def _default_keras_metrics(self) -> Optional[List[Lazy[KerasMetric]]]:
        return [
            (
                keras.metrics.BinaryAccuracy(threshold=0.0, name="accuracy")
                if self.multilabel
                else keras.metrics.SparseCategoricalAccuracy(name="accuracy")
            )
        ]

    def _to_arrays(self, batch: Any, with_targets: bool = True) -> Any:
        """float32 image; ``multi_hot`` for a multi-label target, else int64 ids.

        Keras is **channels-last**: images arrive ``[N, H, W, C]``, so a Keras config's
        preprocessing chain ends after ``ConvertToImage`` rather than adding a CHW ``ToTensor``.
        """
        x = np.asarray(batch_values(batch, self.input_key), dtype="float32")
        if not with_targets:
            return (x,)
        if self.multilabel:
            if not self.num_classes:
                raise RuntimeError(f"{type(self).__name__}: a multi-label target needs num_classes.")
            return x, multi_hot(batch, self.target_key, self.num_classes)
        return x, np.asarray(batch_values(batch, self.target_key), dtype="int64")

    def _as_metric(self, value: Any) -> Any:
        """The TensorFlow loop's metric resolution, with this task's alias vocabulary FIRST.

        A hand-written loop has no ``compile()`` to resolve ``"accuracy"`` contextually, and
        ``keras.metrics.get("accuracy")`` returns the elementwise-equality function — which
        reports 0.0 forever on a raw-logits head. So the alias maps to the same concrete class
        the compile-time resolution would have picked; everything else takes the machinery
        base's generic path (string → ``get``, live instance → per-split clone).
        """
        if isinstance(value, str) and value.lower() in ("accuracy", "acc"):
            return (
                keras.metrics.BinaryAccuracy(threshold=0.0, name="accuracy")
                if self.multilabel
                else keras.metrics.SparseCategoricalAccuracy(name="accuracy")
            )
        parent = super()
        resolved = parent._as_metric(value) if hasattr(parent, "_as_metric") else value
        if callable(resolved) and not hasattr(resolved, "update_state"):
            # A plain function out of `keras.metrics.get` — wrap so the loop can accumulate it.
            return keras.metrics.MeanMetricWrapper(resolved, name=getattr(resolved, "__name__", "metric"))
        return resolved

    # ---- artifacts -------------------------------------------------------------------------- #
    def _before_fit(self) -> None:
        """Persist the run's label vocabulary, BEFORE the fit — see the module docstring.

        Also the more honest place independently of the lifecycle: the vocabulary is a property
        of the DATA, known since ``initialize()``, so a run that crashes mid-fit still records
        what it was training on.
        """
        self._class_names = class_names(self.train_set, self.val_set, self.test_set)
        save_class_names(self._class_names, self._artifact_dir(), self._live_trackers())

    def _artifact_dir(self) -> Optional[str]:
        """Where run artifacts land. This family has no checkpoint-callback convention to read,
        so it follows the sidecar's directory when one is configured."""
        if self.metrics_sidecar_path:
            return str(Path(self.metrics_sidecar_path).expanduser().parent)
        return None

    # ---- predictions ------------------------------------------------------------------------ #
    def _prediction_output(self, preds: Any) -> Any:
        """The batched :class:`~recordstream.ClassificationOutput` — the machinery slices it."""
        logits = np.asarray(preds)
        out: ClassificationOutput[np.ndarray] = {
            "logits": logits,
            "probs": _softmax(logits),
            "class_idx": np.argmax(logits, axis=-1),
        }
        return out


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax. The model emits raw logits (no head activation), like the torch twin."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return cast(np.ndarray, exp / exp.sum(axis=-1, keepdims=True))
