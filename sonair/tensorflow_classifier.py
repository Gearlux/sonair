"""``TensorFlowClassifier`` — the same classification run on a native TensorFlow loop.

The fair question this backend answers: why, when ``KERAS_BACKEND=tensorflow sonair train
--framework keras`` already runs the identical config? Because ``tf.data``, ``tf.GradientTape``
under ``@tf.function`` (which is what makes ``jit_compile``/XLA a knob at all) and the
SavedModel export are TensorFlow's and not Keras's. It is a SIBLING of
:class:`~sonair.keras_classifier.KerasClassifier`, not a subclass — it narrows a precondition
its would-be parent does not have (the engine guard) — and still declares ``framework="keras"``:
the axis names the API a class is WIRED TO.

Since P5 (2026-08-07) this is ``(ClassificationKerasHooks, matrainer.tensorflow.
TensorFlowRunnable)``: the loop machinery — the ``tf.data`` pipeline, the traced steps, the
hand-driven ``CallbackList``, the per-split metric objects, the engine guard, the SavedModel
``_export`` — lives in :class:`matrainer.tensorflow.loop.TensorFlowRunnable`; the task half
(the label pipeline, the multilabel peek, ``num_classes`` resolution) in
:class:`sonair.classification_base.ClassificationBase`; and the task's keras vocabulary (the
``from_logits=True`` branching defaults, the contextual ``"accuracy"`` alias, the
multi-hot/int64 target read, the numpy prediction contract, the before-fit class-names
persistence) in :class:`sonair.classification_keras.ClassificationKerasHooks` — shared
verbatim with the compile/fit sibling, which is the point. What stays here is the FULL flat
constructor.

**Three behaviours the migration NORMALIZED, each a real fix:**

1. **Per-split metric objects are now CLONED** — the old module returned a live wired metric
   as-is, so all three splits shared ONE stateful accumulator (silent cross-contamination; the
   machinery base clones through Keras's own ``get_config``/``from_config``).
2. **The sources are WARMED before a fork** — the old module used a bare
   ``ensure_record_dataset``; the task base's ``_prepare`` now routes every split through
   ``prepare_record_dataset``.
3. **``evaluate()`` scores the VAL split** (the fleet harmonization), and ``computed_metrics``
   raises before ``run()`` like every other backend's.
"""

from typing import Iterable, List, Literal, Optional, Union

from confluid import Lazy, Mandatory, configurable
from loggair import get_logger
from matrainer.metric_names import MetricNames
from matrainer.tracking.base import Tracker
from recordstream import LabelMap, RecordSource

from sonair._tensorflow import keras
from sonair.classification_keras import ClassificationKerasHooks

# AFTER the `_tensorflow` shim import: the shim setdefaults KERAS_BACKEND before anything reads
# it (the matrainer module does the same itself — belt-and-braces, and it keeps this module's
# import story identical to every other sonair TF-touching module).
from matrainer.tensorflow.loop import TensorFlowNetwork, TensorFlowRunnable  # isort: skip

logger = get_logger(__name__)

#: Which entry point :meth:`TensorFlowClassifier.run` dispatches to.
ClassifierTask = Literal["fit", "evaluate", "test", "predict"]

#: A Keras loss / metric / optimizer: a live object, a string alias, or a deferred marker.
KerasLoss = Union[str, keras.losses.Loss]
KerasMetric = Union[str, keras.metrics.Metric]
KerasOptimizer = Union[str, keras.optimizers.Optimizer]

__all__ = [
    "ClassifierTask",
    "KerasLoss",
    "KerasMetric",
    "KerasOptimizer",
    "TensorFlowClassifier",
    "TensorFlowNetwork",
]


@configurable(task="classification", role="trainer", framework="keras")
class TensorFlowClassifier(ClassificationKerasHooks, TensorFlowRunnable):
    """Train, evaluate or predict an image classifier with a native TensorFlow loop.

    Every constructor knob means the same thing as its :class:`~sonair.keras_classifier.
    KerasClassifier` namesake, deliberately: one config drives either backend and only the
    runnable's class changes. The three at the end are what this backend adds.

    Args:
        model: The classification model — wire ``!lazy:sonair.KerasModel`` so ``initialize()``
            can inject the dataset-derived ``num_classes``. ``tf.keras`` IS Keras 3, so the
            Keras backend's wrapper is the wrapper here too — and this backend ALSO takes a
            plain ``tf.Module`` (``TensorFlowModel``, ``TensorFlowConvNet``), which is why the
            slot is typed :data:`TensorFlowNetwork`.
        train_set: Training source (``task="fit"``).
        test_set: Evaluation / prediction source.
        num_classes: Class count. Omitted, it is derived from the data or the label map.
        task: Which entry point ``run()`` dispatches to.
        max_epochs: Training epochs.
        batch_size: Rows per batch.
        trackers: matrainer trackers — one or a list. This run's ONLY output wiring.
        callbacks: ``keras.callbacks.Callback`` instances, driven by hand through a
            ``keras.callbacks.CallbackList`` — this loop has no ``fit()`` to run them.
        experiment_name / run_name: tracking identity.
        input_key / target_key: Record keys for the image and the label.
        label_map: Pin the name-to-id encoding instead of fitting it from the train targets.
        optimizer: Keras optimizer — a string alias (``"adam"``) or an instance. Resolved to an
            INSTANCE before the loop, since there is no ``compile()`` to do it.
        loss: Keras loss. ``None`` picks by target shape (``from_logits=True`` either way — the
            head is a bare ``Dense``).
        metrics: Keras metrics. The bare string ``"accuracy"`` resolves through the task's own
            alias (the hooks module) — a hand-written loop has no ``compile()`` to resolve it
            contextually.
        train_metrics / val_metrics / test_metrics: That split's OWN metrics, merged over
            ``metrics``. This backend narrows properly: each traced step closes over its own
            split's metric objects.
        metric_names: How recorded metrics are named — see
            :class:`~matrainer.metric_names.MetricNames`.
        export_dir: Write a ``tf.saved_model`` directory here after a successful ``fit``.
        jit_compile: Trace the train/eval steps with XLA — opt-in (a real speedup on fixed
            shapes, a real compile cost on the first batch).
        prefetch: Append ``.prefetch(tf.data.AUTOTUNE)`` to every ``tf.data`` pipeline.

    Post-construction slots (YAML / broadcasting): ``val_set``, ``predictions_sink``,
    ``metrics_sidecar_path``.
    """

    def __init__(
        self,
        model: Optional[Mandatory[Lazy[TensorFlowNetwork]]] = None,
        train_set: Optional[Lazy[RecordSource]] = None,
        test_set: Optional[Lazy[RecordSource]] = None,
        num_classes: Optional[int] = None,
        task: ClassifierTask = "fit",
        max_epochs: int = 10,
        batch_size: int = 32,
        trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = None,
        callbacks: Optional[List[Lazy[keras.callbacks.Callback]]] = None,
        experiment_name: str = "sonair_classification",
        run_name: Optional[str] = None,
        input_key: str = "image",
        target_key: str = "class",
        label_map: Optional[LabelMap] = None,
        optimizer: Lazy[KerasOptimizer] = "adam",
        loss: Optional[Lazy[KerasLoss]] = None,
        metrics: Optional[List[Lazy[KerasMetric]]] = None,
        train_metrics: Optional[List[Lazy[KerasMetric]]] = None,
        val_metrics: Optional[List[Lazy[KerasMetric]]] = None,
        test_metrics: Optional[List[Lazy[KerasMetric]]] = None,
        metric_names: Optional[Lazy[MetricNames]] = None,
        export_dir: Optional[str] = None,
        jit_compile: bool = False,
        prefetch: bool = True,
    ) -> None:
        super().__init__(
            model=model,
            train_set=train_set,
            test_set=test_set,
            task=task,
            max_epochs=max_epochs,
            batch_size=batch_size,
            jit_compile=jit_compile,
            prefetch=prefetch,
            export_dir=export_dir,
            trackers=trackers,
            callbacks=callbacks,
            experiment_name=experiment_name,
            run_name=run_name,
            input_key=input_key,
            target_key=target_key,
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            metric_names=metric_names,
        )
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.label_map = label_map
        self.multilabel: bool = False
        # Pinned False for the keras family — a `keras.losses` object takes no torch weight
        # tensor, so the task base must never walk for weights here.
        self.balance_classes: bool = False
        self._class_names: Optional[List[str]] = None
