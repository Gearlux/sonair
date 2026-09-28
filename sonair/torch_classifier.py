"""``TorchClassifier`` — the same classification run as a PLAIN PyTorch training loop.

The control-experiment member of the classification fleet, assembled from the workspace trainer
bases (2026-08-06): the hand-written loop machinery — device bracketing, the epoch driver,
scoring, checkpointing, the plain-function tracking hookup — is
``matrainer.torch.TorchLoopRunnable``, and the classification task — the label pipeline,
``num_classes`` resolution, balancing, the loss/metric defaults — is the SAME
:mod:`sonair.classification_base` + :mod:`sonair.classification_torch` pair the Lightning
backend runs on. That is the control experiment executed structurally: the two backends now
share the task by construction, so "swap the runnable, keep the config" cannot drift. What
remains HERE is the config surface (the full flat constructor — the confluid introspection
reads THIS signature) and the one loop-specific epilogue: persisting the class names beside
the checkpoint after a fit.

Minimal flat YAML — the Lightning block with one key changed::

    runnable: !class:sonair.TorchClassifier
    model: !lazy:sonair.TimmModel { model_name: efficientvit_b0 }
    train_set: !class:recordstream.Stream { source: ..., ops: !ref:preprocess }
    batch_size: 32
    max_epochs: 3

Run it with ``sonair train cfg.yaml --framework torch``.
"""

from typing import Any, Iterable, List, Optional, Union

import torch
import torch.nn as nn
from confluid import Mandatory, Partial, configurable
from loggair import get_logger
from matrainer.artifacts import save_class_names
from matrainer.metric_names import MetricNames
from matrainer.torch import Device, TorchLoopRunnable
from matrainer.torch.metrics import MetricsSpec
from matrainer.tracking.base import Tracker
from recordstream import LabelMap, RecordSource, class_names
from recordstream.runnable import RunnableTask

from sonair.classification_torch import ClassificationTorchHooks

logger = get_logger(__name__)

#: Which entry point :meth:`TorchClassifier.run` dispatches to (the standard four).
ClassifierTask = RunnableTask

__all__ = ["ClassifierTask", "Device", "TorchClassifier"]


@configurable(task="classification", role="trainer", framework="torch")
class TorchClassifier(ClassificationTorchHooks, TorchLoopRunnable):
    """Train, evaluate or predict an image classifier with a hand-written PyTorch loop.

    Args:
        model: Classification model emitting ``[N, num_classes]`` logits — wire ``!lazy:`` so
            :meth:`initialize` can inject the dataset-derived ``num_classes`` at run time.
        train_set: Training source (``task="fit"``).
        test_set: Held-out source (``task="test"`` / ``"predict"``; also tested after a fit).
        num_classes: Class count. Omitted, it is derived from the model or the data.
        task: Which entry point ``run()`` dispatches to.
        max_epochs: Training epochs.
        batch_size: Rows per batch.
        num_workers: DataLoader worker processes. Persistent workers follow from this unless
            ``persistent_workers`` says otherwise.
        persistent_workers: Override the pairing derived from ``num_workers`` (``None`` derives
            ``num_workers != 0``). An explicit value is for a HOST fact no derivation can see —
            macOS terminates persistent workers slowly, so a config there sets ``false`` while
            keeping its workers. ``True`` with ``num_workers=0`` is refused.
        experiment_name: Run identity, recorded in artifacts and tracking.
        run_name: Run identity; a random ``adjective_noun`` when omitted.
        input_key: Record key holding the image.
        target_key: Record key holding the label.
        label_map: Pin the name-to-id encoding instead of fitting it from the train targets.
        balance_classes: Inject inverse-frequency class weights into a weight-capable loss.
        metrics: torchmetrics templates, flowed per split. ``None`` picks by target shape.
        train_metrics / val_metrics / test_metrics: That split's OWN templates, merged over
            ``metrics`` — how a metric is added to one split only (a confusion matrix on
            validation). Each slot holds the spec until :meth:`initialize` builds it, and the
            live prefixed ``MetricCollection`` after.
        plot_metrics: ALSO send each non-scalar metric to the trackers as a matplotlib
            heatmap, not only as a table. Off by default: it makes matplotlib
            (``matrainer[plot]``) a cost of every eval epoch, where the table costs nothing.
        device: Where to run — see :data:`Device`.
        checkpoint_dir: Directory for ``model.pt`` (a plain ``state_dict``) and the run's
            ``class_names.json``. ``None`` writes no weights, which is what a smoke run wants.
        grad_clip: Clip gradients to this global norm before each ``step()``. ``None`` disables.
        trackers: matrainer trackers — one or a list.
        metric_names: How recorded metrics are named — see
            :class:`~matrainer.metric_names.MetricNames`. This loop already writes the canonical
            ``<split>/<name>``, so the default translates nothing; wiring ``target:`` is what
            makes it record in another engine's spelling instead.

    Optional OVERRIDE slots — every one is already constructed with a sensible default in
    ``__init__`` (the task-shaped ones, ``loss_fn`` + the metric slots, resolve to this task's
    defaults in ``initialize()``), so nothing here NEEDS wiring; a same-named top-level YAML key
    replaces one via broadcasting when you want to: ``val_set``, ``optimizer``,
    ``train/val/test_loader``, ``loss_fn``, ``predictions_sink``, ``metrics_sidecar_path``.
    """

    def __init__(
        self,
        model: Optional[Mandatory[Partial[nn.Module]]] = None,
        train_set: Optional[Partial[RecordSource]] = None,
        test_set: Optional[Partial[RecordSource]] = None,
        num_classes: Optional[int] = None,
        task: ClassifierTask = "fit",
        max_epochs: int = 10,
        batch_size: int = 32,
        num_workers: int = 0,
        persistent_workers: Optional[bool] = None,
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
        device: Device = "auto",
        checkpoint_dir: Optional[str] = None,
        grad_clip: Optional[float] = None,
        trackers: Optional[Union[Partial[Tracker], Iterable[Partial[Tracker]]]] = None,
        metric_names: Optional[Partial[MetricNames]] = None,
    ) -> None:
        super().__init__(
            model=model,
            train_set=train_set,
            test_set=test_set,
            task=task,
            max_epochs=max_epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
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
            device=device,
            checkpoint_dir=checkpoint_dir,
            grad_clip=grad_clip,
        )
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.label_map = label_map
        self.balance_classes = balance_classes
        self.class_weights: Optional[torch.Tensor] = None
        self.multilabel: bool = False
        self.plot_metrics = plot_metrics
        self._class_names: Optional[List[str]] = None

    def _after_fit(self, trainer: Any, train_ds: Any, val_ds: Any, test_ds: Any) -> None:
        """The loop epilogue: persist the run's class names beside the checkpoint.

        The vocabulary is read off the encoded splits (``recordstream.class_names`` skips
        ``None`` splits) and lands via matrainer's artifact helper — which also logs it to
        every wired tracker — in the same directory as ``model.pt``, so a later
        evaluate/predict run finds it knowing only the checkpoint path.
        """
        self._class_names = class_names(train_ds, val_ds, test_ds)
        save_class_names(self._class_names, self.checkpoint_dir, self._live_trackers())
