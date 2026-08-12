"""``KerasClassifier`` — the same classification run, on Keras 3 instead of Lightning.

Since P5 (2026-08-07) this is ``(ClassificationKerasHooks, matrainer.keras.KerasRunnable)``:
the compile/fit machinery — build + compile, the ``History`` read-back, the entry points, the
``RecordSequence`` batch boundary, the tracking callback — lives in
:class:`matrainer.keras.trainer.KerasRunnable`; the task half (the multilabel peek, the
``LabelMap`` pipeline, ``num_classes`` resolution, the single announced target walk) in
:class:`sonair.classification_base.ClassificationBase`; and the task's KERAS VOCABULARY (the
``from_logits=True`` branching defaults, the ``accuracy``-named metric objects, the
multi-hot/int64 target read, the numpy prediction contract, the before-fit class-names
persistence) in :class:`sonair.classification_keras.ClassificationKerasHooks` — written once
for this concrete AND its native-TensorFlow sibling. What stays here is the FULL flat
constructor (the confluid introspection surface).

**Two real differences from the torch backends, both about the ENGINE:**

1. **Keras is channels-LAST**, so a Keras config's preprocessing chain ends after
   ``ConvertToImage`` rather than adding the CHW ``ToTensor`` the torch chain ends with.
2. **``from_logits=True`` is mandatory on the loss** — see the hooks module for the measured
   failure mode (a finite loss that simply does not learn).

``metrics`` / ``train_metrics`` / ``val_metrics`` / ``test_metrics`` are UNIONED (the family's
``_compiled_metrics``): ``keras.Model.compile`` takes ONE list — see the machinery base. The
migration also gained this backend the fleet's fork-safety warm (the task base's ``_prepare``
routes every split through ``prepare_record_dataset`` — the old module used a bare
``ensure_record_dataset`` and never warmed).

Keras 3 picks its engine from ``KERAS_BACKEND``; this module never sets it.
"""

from typing import Iterable, List, Literal, Optional, Union

from confluid import Mandatory, Partial, configurable
from loggair import get_logger
from matrainer.keras.runnable import KerasLoss, KerasMetric, KerasOptimizer
from matrainer.keras.trainer import KerasRunnable
from matrainer.metric_names import MetricNames
from matrainer.tracking.base import Tracker
from recordstream import LabelMap, RecordSource
from recordstream.keras import keras

from sonair.classification_keras import ClassificationKerasHooks

logger = get_logger(__name__)

#: Which entry point :meth:`KerasClassifier.run` dispatches to.
ClassifierTask = Literal["fit", "evaluate", "test", "predict"]

__all__ = ["ClassifierTask", "KerasClassifier", "KerasLoss", "KerasMetric", "KerasOptimizer"]


@configurable(task="classification", role="trainer", framework="keras")
class KerasClassifier(ClassificationKerasHooks, KerasRunnable):
    """Train, evaluate or predict an image classifier with Keras 3.

    Args:
        model: The classification model — wire ``!lazy:`` so ``initialize()`` can inject the
            dataset-derived ``num_classes``.
        train_set: Training source (``task="fit"``).
        test_set: Evaluation / prediction source.
        num_classes: Class count. Omitted, it is derived from the data or the label map.
        task: Which entry point ``run()`` dispatches to.
        max_epochs: Training epochs.
        batch_size: Rows per batch.
        trackers: matrainer trackers — one or a list. This run's ONLY output wiring.
        callbacks: ``keras.callbacks.Callback`` instances (``ModelCheckpoint``,
            ``EarlyStopping`` …). A file-writing one gets its parent directory created.
        experiment_name / run_name: tracking identity.
        input_key / target_key: Record keys for the image and the label.
        label_map: Pin the name-to-id encoding instead of fitting it from the train targets.
        optimizer: Keras optimizer — a string alias (``"adam"``) or an instance.
        loss: Keras loss (the registered set lives in :mod:`sonair.keras_classification`).
            ``None`` picks by target shape, like the Lightning backend's sentinel.
        metrics / train_metrics / val_metrics / test_metrics: Keras metrics. This engine
            computes their UNION — see the machinery base.
        metric_names: How recorded metrics are named — see
            :class:`~matrainer.metric_names.MetricNames`. Keras writes ``loss`` / ``val_loss``;
            the default translates that to the canonical ``train/loss`` / ``val/loss`` on the
            way to the trackers and the sidecar. What Keras hands its OWN callbacks is
            untouched — ``EarlyStopping`` monitors ``val_loss`` and must keep finding it.

    Post-construction slots (YAML / broadcasting): ``val_set``, ``predictions_sink``,
    ``metrics_sidecar_path``.
    """

    def __init__(
        self,
        model: Optional[Mandatory[Partial[keras.Model]]] = None,
        train_set: Optional[Partial[RecordSource]] = None,
        test_set: Optional[Partial[RecordSource]] = None,
        num_classes: Optional[int] = None,
        task: ClassifierTask = "fit",
        max_epochs: int = 10,
        batch_size: int = 32,
        trackers: Optional[Union[Partial[Tracker], Iterable[Partial[Tracker]]]] = None,
        callbacks: Optional[List[Partial[keras.callbacks.Callback]]] = None,
        experiment_name: str = "sonair_classification",
        run_name: Optional[str] = None,
        input_key: str = "image",
        target_key: str = "class",
        label_map: Optional[LabelMap] = None,
        optimizer: Partial[KerasOptimizer] = "adam",
        loss: Optional[Partial[KerasLoss]] = None,
        metrics: Optional[List[Partial[KerasMetric]]] = None,
        train_metrics: Optional[List[Partial[KerasMetric]]] = None,
        val_metrics: Optional[List[Partial[KerasMetric]]] = None,
        test_metrics: Optional[List[Partial[KerasMetric]]] = None,
        metric_names: Optional[Partial[MetricNames]] = None,
    ) -> None:
        super().__init__(
            model=model,
            train_set=train_set,
            test_set=test_set,
            task=task,
            max_epochs=max_epochs,
            batch_size=batch_size,
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
        # The keras family answers the balancing hooks with the base's safe default (False /
        # no-op): a `keras.losses` object takes no torch weight tensor, so balancing is
        # deliberately not served here — pinned False so the task base never walks for weights.
        self.balance_classes: bool = False
        self._class_names: Optional[List[str]] = None
