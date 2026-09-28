"""``FastaiClassifier`` — the same classification run, driven by a fastai ``Learner``.

The fastai member of the classification fleet, assembled from the workspace trainer bases
(2026-08-07): the Learner machinery — the ``DataLoaders``/``Learner`` assembly, the positional
``_to_tuple`` collate (built on the task's own ``_targets`` hook), the Recorder read-back into
canonical names, the plain-function tracking hookup — is ``matrainer.fastai.FastaiRunnable``,
and the classification task — the label pipeline, ``num_classes`` resolution, the multilabel
peek — is the SAME :mod:`sonair.classification_base` + :mod:`sonair.classification_torch` pair
every torch-family backend runs on. What remains HERE is the config surface (the full flat
constructor — the confluid introspection reads THIS signature), the task's fastai VOCABULARY
(the ``accuracy``/``accuracy_multi`` defaults, the ``*Flat`` loss pair, the ``f1_*`` configured
aliases), and the classification epilogue persisting the class names.

**Metrics are named as STRINGS here, and that is a fastai fact, not a sonair shortcut.** Nearly
every fastai classification metric is a plain *function* (``accuracy``, ``error_rate``,
``top_k_accuracy``) or a factory that returns one (``F1Score()``, ``RocAuc()``), so there is no
class for a ``!class:`` tag to name and nothing for confluid's registry to tag::

    metrics: [accuracy, top_k_accuracy]

Run it with ``sonair train cfg.yaml --framework fastai``.
"""

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

import torch
from confluid import Mandatory, Partial, configurable
from fastai import metrics as fastai_metrics
from fastai.callback.core import Callback as FastaiCallback
from fastai.losses import BCEWithLogitsLossFlat, CrossEntropyLossFlat
from loggair import get_logger
from matrainer.artifacts import save_class_names
from matrainer.fastai import FastaiLoss, FastaiMetric, FastaiOptimizer, FastaiRunnable
from matrainer.metric_names import MetricNames
from matrainer.tracking.base import Tracker
from recordstream import LabelMap, RecordSource, class_names
from recordstream.runnable import RunnableTask

from sonair.classification_torch import ClassificationTorchHooks

logger = get_logger(__name__)

#: Which entry point :meth:`FastaiClassifier.run` dispatches to (the standard four).
ClassifierTask = RunnableTask

__all__ = ["ClassifierTask", "FastaiClassifier"]


@configurable(task="classification", role="trainer", framework="torch")
class FastaiClassifier(ClassificationTorchHooks, FastaiRunnable):
    """Train, evaluate or predict an image classifier with a fastai ``Learner``.

    Args:
        model: The classification model — wire ``!lazy:sonair.TimmModel`` so ``initialize()``
            can inject the dataset-derived ``num_classes``. fastai is torch, so this is the
            reference backend's wrapper, unchanged.
        train_set: Training source (``task="fit"``).
        val_set: Validation source. fastai's ``Learner`` always validates, so one is used
            whether or not it is wired.
        test_set: Evaluation / prediction source.
        num_classes: Class count. Omitted, it is derived from the data or the label map.
        task: Which entry point ``run()`` dispatches to.
        max_epochs: Training epochs.
        batch_size: Rows per batch.
        num_workers: DataLoader worker processes.
        persistent_workers: Override the pairing derived from ``num_workers`` (``None`` derives
            ``num_workers != 0``). An explicit value is for a HOST fact no derivation can see —
            macOS terminates persistent workers slowly, so a config there sets ``false`` while
            keeping its workers. ``True`` with ``num_workers=0`` is refused.
        trackers: matrainer trackers — one or a list. This run's ONLY output wiring:
            metrics, provenance and the run manifest all go to them.
        callbacks: ``fastai.callback.core.Callback`` instances (``CSVLogger``,
            ``SaveModelCallback`` …), appended to the ``Learner``'s own set.
        experiment_name: Run identity, recorded in artifacts and tracking.
        run_name: Run identity; a random ``adjective_noun`` when omitted.
        input_key: Record key holding the image.
        target_key: Record key holding the label.
        label_map: Pin the name-to-id encoding instead of fitting it from the train targets.
        learning_rate: Peak LR for :meth:`fit` (``lr_max`` under one-cycle, ``lr`` otherwise).
        one_cycle: Use ``fit_one_cycle`` (fastai's default schedule) rather than a flat
            ``fit``. This is fastai's headline training behaviour, so it is on by default.
        weight_decay: fastai's ``wd``. ``None`` leaves the ``Learner`` default.
        opt_func: A ``fastai.optimizer`` function or its name (``"Adam"``, ``"SGD"``,
            ``"ranger"``, …).
        loss: A fastai / torch loss callable. ``None`` picks by target shape, like every other
            backend's sentinel — the ``*Flat`` pair, which for this task behaves exactly as
            ``nn.CrossEntropyLoss`` / ``nn.BCEWithLogitsLoss`` do (logits in, no head
            activation).
        metrics: fastai metrics — callables, names resolved against ``fastai.metrics``, or one
            of this backend's configured aliases (``f1_macro`` / ``f1_micro`` / ``f1_weighted``,
            which name a particular ``F1Score(average=…)`` the module exposes only as a
            factory). ``None`` picks ``accuracy`` / ``accuracy_multi`` by target shape.
        train_metrics / val_metrics / test_metrics: That split's OWN metrics. ``test_metrics``
            merges over ``metrics`` and narrows; ``train_metrics`` and ``val_metrics`` are
            UNIONED, because a fitting ``Learner`` reports one list on both passes.
        metric_names: How recorded metrics are named — see
            :class:`~matrainer.metric_names.MetricNames`. The ``Recorder``'s columns are
            ``train_loss`` / ``valid_loss`` / the metrics; the default translates them to the
            canonical ``train/loss`` / ``val/loss`` / ``val/<metric>``, an unprefixed column
            landing under ``val`` because that is the split a ``Learner`` measures.

    Optional OVERRIDE slots — every one is already constructed with a sensible default in
    ``__init__``, so nothing here NEEDS wiring; a same-named top-level YAML key replaces one
    via broadcasting when you want to: ``train/val/test_loader`` (the torch ``DataLoader``s),
    ``dataloaders`` (fastai's ``DataLoaders`` around them — its ``path``), ``learner`` (every
    ``Learner`` knob this class does not restate: ``splitter``, ``model_dir``, ``moms``,
    ``wd_bn_bias``, ``train_bn``, ``default_cbs``), ``predictions_sink``,
    ``metrics_sidecar_path``.
    """

    def __init__(
        self,
        model: Optional[Mandatory[Partial[torch.nn.Module]]] = None,
        train_set: Optional[Partial[RecordSource]] = None,
        val_set: Optional[Partial[RecordSource]] = None,
        test_set: Optional[Partial[RecordSource]] = None,
        num_classes: Optional[int] = None,
        task: ClassifierTask = "fit",
        max_epochs: int = 10,
        batch_size: int = 32,
        num_workers: int = 0,
        persistent_workers: Optional[bool] = None,
        trackers: Optional[Union[Partial[Tracker], Iterable[Partial[Tracker]]]] = None,
        callbacks: Optional[List[Partial[FastaiCallback]]] = None,
        experiment_name: str = "sonair_classification",
        run_name: Optional[str] = None,
        input_key: str = "image",
        target_key: str = "class",
        label_map: Optional[LabelMap] = None,
        learning_rate: float = 1e-3,
        one_cycle: bool = True,
        weight_decay: Optional[float] = None,
        opt_func: FastaiOptimizer = "Adam",
        loss: Optional[Partial[FastaiLoss]] = None,
        metrics: Optional[Sequence[FastaiMetric]] = None,
        train_metrics: Optional[Sequence[FastaiMetric]] = None,
        val_metrics: Optional[Sequence[FastaiMetric]] = None,
        test_metrics: Optional[Sequence[FastaiMetric]] = None,
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
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            one_cycle=one_cycle,
            opt_func=opt_func,
            loss=loss,
            callbacks=callbacks,
        )
        self.val_set = val_set
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.label_map = label_map
        # The fastai defaults are the `*Flat` family, none of which declares a `weight` ctor
        # param — so class balancing was never served on this backend, and the task base's
        # probe is switched off rather than left to answer "no" per run.
        self.balance_classes = False
        self.class_weights: Optional[torch.Tensor] = None
        self.multilabel: bool = False
        self.plot_metrics = False
        self._class_names: Optional[List[str]] = None

    # ---- the task's fastai vocabulary ------------------------------------------------------- #
    def _default_fastai_loss(self) -> Any:
        """The ``*Flat`` pair — fastai's own spelling of the task hook's CE/BCE sentinel.

        They flatten the trailing dimensions before delegating to the torch loss underneath;
        for this task that behaves exactly as ``nn.CrossEntropyLoss`` / ``nn.BCEWithLogitsLoss``
        do (logits in, no head activation), so the choice matches the other backends' sentinel
        pair rather than diverging from it.
        """
        return BCEWithLogitsLossFlat() if self.multilabel else CrossEntropyLossFlat()

    def _default_fastai_metrics(self) -> List[FastaiMetric]:
        return ["accuracy_multi"] if self.multilabel else ["accuracy"]

    def _metric_aliases(self) -> Dict[str, Callable[[], Any]]:
        """The ``f1_*`` names — a particular ``F1Score(average=…)`` the module exposes only as
        a FACTORY.

        ``fastai.metrics`` has no ``f1_macro`` attribute at all, and a bare
        ``getattr(fastai_metrics, "F1Score")`` would hand the LEARNER a factory rather than a
        metric — accepted without complaint and then reported under fastai's own ``f1_score``
        column with the default ``average="binary"``, which on a 10-class problem raises deep
        inside scikit-learn. Naming the configured form is the only spelling a config can get
        right.
        """
        return {
            "f1_macro": lambda: _built_metric("macro"),
            "f1_micro": lambda: _built_metric("micro"),
            "f1_weighted": lambda: _built_metric("weighted"),
        }

    def _metric_vocabulary_hint(self) -> str:
        return (
            "For classification: accuracy, accuracy_multi, error_rate, top_k_accuracy, "
            "BalancedAccuracy, MatthewsCorrCoef."
        )

    def _after_fit(self, trainer: Any, train_ds: Any, val_ds: Any, test_ds: Any) -> None:
        """The classification epilogue: persist the run's class names beside the artifacts.

        The vocabulary is read off the encoded splits (``recordstream.class_names`` skips
        ``None`` splits) and lands via matrainer's artifact helper — which also logs it to
        every wired tracker — so a later evaluate/predict run finds it knowing only the
        artifact path.
        """
        self._class_names = class_names(train_ds, val_ds, test_ds)
        save_class_names(self._class_names, self._artifact_dir(), self._live_trackers())


def _built_metric(average: str) -> Any:
    """``fastai.metrics.F1Score(average=…)``, renamed to the workspace-canonical column."""
    metric = fastai_metrics.F1Score(average=average)
    # `AccumMetric.name` is what the Recorder writes as the COLUMN, and the factory derives it
    # from the wrapped function: `f1_score` for every averaging mode. Overriding it is what
    # lines this run's number up with the torch backends' `f1_macro` instead of opening a
    # second column for the same quantity (the ONE NAME PER QUANTITY mandate).
    metric.name = f"f1_{average}"
    return metric
