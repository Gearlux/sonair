"""LightningClassifier: a standard PyTorch-Lightning image classifier (train + eval in one class).

The Lightning member of the classification fleet, assembled from the workspace trainer bases
(2026-08-06): the Lightning machinery — loaders, the deferred ``pl.Trainer`` slot, the step
functions, tracking — is ``matrainer.lightning.LightningRunnable``, and the classification task
— the label pipeline, ``num_classes`` resolution, balancing, the loss/metric defaults — is
:mod:`sonair.classification_base` + :mod:`sonair.classification_torch`. What remains HERE is
the config surface (the full flat constructor — the confluid introspection reads THIS
signature) and the one Lightning-specific epilogue: persisting the class names beside the
checkpoints after a fit.

Minimal flat YAML (top-level keys wired by Confluid broadcasting)::

    runnable: !class:sonair.LightningClassifier
      task: fit
      run_name: efficientvit_b0
    model: !lazy:sonair.TimmModel { model_name: efficientvit_b0, pretrained: true }
    train_set: !class:recordstream.sources.huggingface.HuggingFaceSource { path: kitofrank/RFUAV, split: train }
    val_set: !class:recordstream.sources.huggingface.HuggingFaceSource { path: kitofrank/RFUAV, split: validation }
    batch_size: 32
    max_epochs: 20

    recordstream run train.yaml
"""

from typing import Any, Iterable, List, Optional, Union

import pytorch_lightning as L
import torch
import torch.nn as nn
from confluid import Lazy, Mandatory, configurable
from loggair import get_logger
from matrainer.artifacts import save_class_names
from matrainer.lightning import LightningRunnable, checkpoint_dir
from matrainer.metric_names import MetricNames
from matrainer.torch.metrics import MetricsSpec
from matrainer.tracking.base import Tracker
from recordstream import LabelMap, RecordSource, class_names
from recordstream.runnable import RunnableTask

from sonair.classification_torch import ClassificationTorchHooks

logger = get_logger(__name__)

#: Which entry point :meth:`LightningClassifier.run` dispatches to (the standard four).
ClassifierTask = RunnableTask


@configurable(task="classification", role="trainer", framework="torch")
class LightningClassifier(ClassificationTorchHooks, LightningRunnable):
    """A standard Lightning image classifier — train, evaluate, test, or predict via ``task``.

    Args:
        model: Classification model emitting ``[N, num_classes]`` logits (wire ``!lazy:`` so
            :meth:`initialize` flows it with the resolved num_classes at run time).
        train_set: Training dataset / Stream (for ``task="fit"``).
        test_set: Held-out eval dataset / Stream (for ``task="test"`` / ``"predict"``; also tested
            after ``fit`` when set).
        num_classes: Output class count; omit to resolve from the model or the dataset.
        task: Entry point :meth:`run` dispatches to — ``"fit"`` / ``"evaluate"`` / ``"test"`` / ``"predict"``.
        max_epochs / batch_size / num_workers: ergonomic flat-config knobs.
        trackers: matrainer trackers — one or a list. They ARE this run's metric sinks:
            each is adapted to Lightning's logger interface, so there is no separate
            `logger:` slot of `pytorch_lightning.loggers`.
        experiment_name / run_name: tracking identity.
        input_key: Record key carrying the model input image (default ``"image"``).
        target_key: Record key carrying the classification target (default ``"class"``).
        label_map: Optional pinned name→id :class:`~recordstream.LabelMap` (fitted from the train set
            for string targets when omitted; reloaded from a checkpoint-adjacent ``class_names.json``
            on the eval path).
        balance_classes: Inject inverse-frequency class weights into a weight-capable loss (``fit``).
        metrics: Single-metric templates flowed per split (default top-1 accuracy + macro F1).
        train_metrics / val_metrics / test_metrics: That split's OWN templates, merged over
            ``metrics`` — how a metric is added to one split only (a confusion matrix on
            validation). Each slot holds the spec until :meth:`initialize` builds it, and the
            live prefixed ``MetricCollection`` after.
        plot_metrics: ALSO send each non-scalar metric to the trackers as a matplotlib
            heatmap, not only as a table. Off by default: it makes matplotlib
            (``matrainer[plot]``) a cost of every eval epoch, where the table costs nothing.
        metric_names: How recorded metrics are named — see
            :class:`~matrainer.metric_names.MetricNames`. The ``self.log`` calls below already
            write the canonical ``<split>/<name>``, so the default translates nothing; it is
            handed to each ``TrackerLogger``, where a wired ``target:`` records the same run in
            another engine's spelling.

    Optional OVERRIDE slots — every one is already constructed with a sensible default in
    ``__init__`` (the task-shaped ones, ``loss_fn`` + the metric slots, resolve to this task's
    defaults in ``initialize()``), so nothing here NEEDS wiring; a same-named top-level YAML key
    replaces one via broadcasting when you want to. ``val_set`` is the one a config typically
    does wire (there is no sensible default validation split): ``val_set``, ``optimizer``,
    ``train/val/test_loader``, ``lightning``, ``loss_fn``, ``predictions_sink``,
    ``metrics_sidecar_path``, ``log_system_metrics``.
    """

    def __init__(
        self,
        model: Mandatory[Lazy[nn.Module]],
        train_set: Optional[Lazy[RecordSource]] = None,
        test_set: Optional[Lazy[RecordSource]] = None,
        num_classes: Optional[int] = None,
        task: ClassifierTask = "fit",
        max_epochs: int = 10,
        batch_size: int = 32,
        num_workers: int = 0,
        trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = None,
        experiment_name: str = "sonair_classification",
        run_name: Optional[str] = None,
        input_key: str = "image",
        target_key: str = "class",
        label_map: Optional[LabelMap] = None,
        balance_classes: bool = True,
        metrics: MetricsSpec = None,
        train_metrics: MetricsSpec = None,
        val_metrics: MetricsSpec = None,
        test_metrics: MetricsSpec = None,
        plot_metrics: bool = False,
        metric_names: Optional[Lazy[MetricNames]] = None,
    ) -> None:
        super().__init__(
            model=model,
            train_set=train_set,
            test_set=test_set,
            task=task,
            max_epochs=max_epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            trackers=trackers,
            experiment_name=experiment_name,
            run_name=run_name,
            input_key=input_key,
            target_key=target_key,
            metrics=metrics,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            metric_names=metric_names,
        )
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.label_map = label_map
        self.balance_classes = balance_classes
        self.class_weights: Optional[torch.Tensor] = None
        self.multilabel: bool = False
        self.plot_metrics = plot_metrics
        self._class_names: Optional[List[str]] = None

    def _after_fit(self, trainer: L.Trainer, train_ds: Any, val_ds: Any, test_ds: Any) -> None:
        """The Lightning epilogue: persist the run's class names beside the checkpoints.

        The vocabulary is read off the encoded splits (``recordstream.class_names`` skips
        ``None`` splits) and lands via matrainer's artifact helper — which also logs it to
        every wired tracker — in the directory Lightning checkpointed to, so a later
        evaluate/predict run finds it knowing only the checkpoint path.
        """
        self._class_names = class_names(train_ds, val_ds, test_ds)
        save_class_names(self._class_names, checkpoint_dir(trainer), self._live_trackers())


__all__ = ["ClassifierTask", "LightningClassifier"]
