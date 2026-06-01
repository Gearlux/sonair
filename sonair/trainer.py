"""LightningClassificationTrainer: specialised image-classification trainer.

Subclass of :class:`~marainer.trainer.LightningTrainer` that bakes in all
classification-specific defaults so a minimal YAML only needs the dataset and
the model backbone — no adapter wiring for loss, collate, metrics, loaders, or
the Lightning Trainer, and no ``num_classes`` (it is derived from the dataset).

Baked-in defaults (all overridable via YAML ``@configurable`` attributes):

* ``loss_fn`` — :class:`torch.nn.CrossEntropyLoss` (override via the ``loss_fn``
  slot; any ``task="classification", role="loss"`` class is offered)
* ``train/val/test_loader`` — built from the ``batch_size`` / ``num_workers`` /
  ``persist_workers`` knobs, with :func:`~sonair.classification.classification_collate_fn`
* ``optimizer`` — ``AdamW(lr=1e-4, weight_decay=0.05)``
* ``lightning`` — ``pytorch_lightning.Trainer`` built from ``max_epochs`` + ``logger``
* Metrics — :class:`~torchmetrics.classification.MulticlassAccuracy` (top-1)
  + :class:`~torchmetrics.classification.MulticlassF1Score` (macro) per split;
  top-5 accuracy added when ``num_classes >= 5``.

Minimal **flat** YAML — top-level keys are wired into the trainer by Confluid
broadcasting (no nesting / ``!ref`` needed); the ``!lazy:`` model gets
``num_classes`` injected from the dataset at run time::

    trainer: !class:sonair.LightningClassificationTrainer
      run_name: efficientvit_b0
    model: !lazy:sonair.TimmModel
      model_name: efficientvit_b0
      pretrained: true
    train_set: !class:dataflux.sources.HuggingFaceSource
      path: kitofrank/RFUAV
      split: train
    val_set: !class:dataflux.sources.HuggingFaceSource
      path: kitofrank/RFUAV
      split: validation
    batch_size: 32
    num_workers: 8
    max_epochs: 20
"""

from typing import Any, List, Optional

import pytorch_lightning as L
import torch
import torch.nn as nn
from confluid import (  # LazyClass is the deferred Fluid (confluid.Lazy is the annotation)
    Class,
    LazyClass,
    configurable,
    flow,
)
from confluid.fluid import Fluid
from marainer.trainer import LightningTrainer
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from sonair.classification import classification_collate_fn


def _build_classification_metrics(num_classes: int) -> MetricCollection:
    metrics: dict[str, Any] = {
        "accuracy_top1": MulticlassAccuracy(num_classes=num_classes, top_k=1),
        "f1_macro": MulticlassF1Score(num_classes=num_classes, average="macro"),
    }
    if num_classes >= 5:
        metrics["accuracy_top5"] = MulticlassAccuracy(num_classes=num_classes, top_k=5)
    return MetricCollection(metrics)


def _dataset_num_classes(dataset: Any) -> Optional[int]:
    """Read ``num_classes`` from a dataset's lazy hook (``num_classes()`` method,
    else ``derived_dimensions()["num_classes"]``); ``None`` if it exposes neither."""
    nc = getattr(dataset, "num_classes", None)
    if callable(nc):
        return int(nc())
    derive = getattr(dataset, "derived_dimensions", None)
    if callable(derive):
        value = derive().get("num_classes")
        return int(value) if value is not None else None
    return None


@configurable(task="classification", role="trainer")
class LightningClassificationTrainer(LightningTrainer):
    """Specialised image-classification trainer with baked-in defaults.

    Mirrors :class:`~marainer.trainer.LightningTrainer`'s constructor discipline
    (confluid "Lazy Initialization & Zero-Arg Construction"): the constructor
    only takes ``model`` / ``train_set`` (+ identity ``num_classes`` /
    ``experiment_name`` / ``run_name``) and does **no functional work** — every
    other slot is a defaulted attribute set in the body and configured
    post-construction (by YAML, or by Confluid broadcasting of flat top-level
    keys). Classification baked-in attributes (all overridable post-construction):

    * ``loss_fn`` — :class:`torch.nn.CrossEntropyLoss`
    * ``optimizer`` — deferred ``AdamW(lr=1e-4, weight_decay=0.05)``
    * ``train/val/test_loader`` — deferred ``DataLoader`` with
      :func:`~sonair.classification.classification_collate_fn`
    * Metrics — top-1 ``MulticlassAccuracy`` + macro ``MulticlassF1Score``
      (+ top-5 when ``num_classes >= 5``); built in :meth:`run` once ``num_classes``
      is resolved. **sonair OWNS the num_classes derivation** (:meth:`_resolve_model_and_metrics`):
      it resolves the count (explicit trainer arg → explicit on the model config →
      derived from the dataset) and flows + realizes the model with it directly,
      rather than relying on the generic ``marainer.dimensions.flow_node_with_dims``.

    **Ergonomic flat-config knobs** — ``batch_size`` / ``num_workers`` /
    ``persist_workers`` / ``max_epochs`` / ``lightning_loggers`` are plain
    defaulted attributes too, so Confluid broadcasting injects matching flat
    top-level YAML keys into them. They are folded into the deferred loader /
    Trainer stubs in :meth:`run` (NOT in ``__init__`` — run-time work keeps the
    constructor clean). (``lightning_loggers`` is named to dodge the read-only
    ``LightningModule.logger`` / ``.loggers`` properties; it becomes ``L.Trainer(logger=…)``.)

    Args:
        model: Classification model emitting ``[N, num_classes]`` logits (wire ``!lazy:`` so the
            trainer flows it with the resolved num_classes at run() — see :meth:`_resolve_model_and_metrics`).
        train_set: Training dataset or Flux pipeline.
        num_classes: Output class count; omit to derive from the dataset (or the model config) at run().
        batch_size: DataLoader batch size folded into the default loaders at run().
        num_workers: DataLoader worker-process count folded into the default loaders at run().
        persist_workers: Keep DataLoader workers alive between epochs (only honored when ``num_workers > 0``).
        max_epochs: ``pytorch_lightning.Trainer(max_epochs=...)`` for the default Trainer built at run().
        lightning_loggers: Optional list of Lightning loggers (``List``-typed so a flat
            ``lightning_loggers: [...]`` broadcasts in); become ``L.Trainer(logger=[...])``.
        experiment_name: MLflow experiment name.
        run_name: MLflow run name.
    """

    def __init__(
        self,
        model: Any,
        train_set: Any,
        num_classes: Optional[int] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        persist_workers: bool = False,
        max_epochs: int = 10,
        lightning_loggers: Optional[List[Any]] = None,
        experiment_name: str = "sonair_classification",
        run_name: str = "train",
    ) -> None:
        super().__init__(model=model, train_set=train_set, experiment_name=experiment_name, run_name=run_name)
        # Store config ONLY — no functional work. The loaders/optimizer are deferred
        # (LazyClass); the base run() flows them (loaders with the dataset, optimizer
        # with model params). The ergonomic knobs are folded into the loader/Trainer
        # stubs at run() time (not here). `lightning_loggers` is a List-typed param so
        # Confluid broadcasts a flat `lightning_loggers: [...]` list into it (and it
        # dodges the read-only LightningModule.logger/.loggers properties).
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persist_workers = persist_workers
        self.max_epochs = max_epochs
        self.lightning_loggers = lightning_loggers
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = LazyClass(torch.optim.AdamW, lr=1e-4, weight_decay=0.05)
        self.train_loader = LazyClass(DataLoader, collate_fn=classification_collate_fn)
        self.val_loader = LazyClass(DataLoader, collate_fn=classification_collate_fn)
        self.test_loader = LazyClass(DataLoader, collate_fn=classification_collate_fn)
        # `lightning` stays the base default LazyClass(L.Trainer); knobs folded in run().
        # Metrics: built eagerly here when num_classes is given (cheap; lets callers inspect them
        # post-construction). When omitted they stay None and run() builds them once
        # _resolve_model_and_metrics() derives the count. Either way sonair OWNS the num_classes
        # resolution (explicit -> model-config -> dataset) — NOT the generic marainer.dimensions path.
        if self.num_classes is not None:
            base = _build_classification_metrics(self.num_classes)
            self.train_metrics = base.clone(prefix="train/")
            self.val_metrics = base.clone(prefix="val/")
            self.test_metrics = base.clone(prefix="test/")

    def run(self) -> None:
        """Fold the ergonomic knobs into the deferred stubs, then run the base flow.

        Knob application is RUN-TIME work (the constructor stays side-effect-free).
        It runs after liquifai's deep-flow, so building a concrete ``Class`` for
        ``lightning`` here is safe (deep-flow never sees it).
        """
        self._apply_loader_knobs()
        self._apply_lightning_knobs()
        self._resolve_model_and_metrics()
        super().run()

    def _resolve_model_and_metrics(self) -> None:
        """Determine ``num_classes`` and build the model + metrics with it — sonair OWNS
        this (NOT the generic ``marainer.dimensions.flow_node_with_dims``).

        Resolution order: an explicit trainer ``num_classes`` → an explicit
        ``num_classes`` on the model's deferred config (a config value wins) → derived
        from the training dataset (walks its targets). The model is then flowed with the
        resolved count and **realized** so its parameters exist before Lightning's
        ``configure_optimizers``; the metric set is built eagerly. By the time the base
        ``run()`` executes, the model + metrics are LIVE, so its generic
        dimension-derivation (`flow_node_with_dims`) no-ops.
        """
        train_ds = flow(self.train_set)
        self.train_set = train_ds  # reuse — the base run() re-flows idempotently (no double load)
        num_classes = self.num_classes
        if num_classes is None and isinstance(self.model, Fluid):
            num_classes = self.model.kwargs.get("num_classes")
        if num_classes is None:
            num_classes = _dataset_num_classes(train_ds)
        if not num_classes:
            raise RuntimeError(
                "LightningClassificationTrainer: could not determine num_classes — set it on the "
                "trainer or the model, or wire a dataset exposing num_classes()/derived_dimensions()."
            )
        self.num_classes = int(num_classes)
        # Flow the model with the resolved count (runtime kwarg wins → model + metrics agree).
        if isinstance(self.model, Fluid):
            self.model = flow(self.model, num_classes=self.num_classes)
        else:
            self.model = flow(self.model)
        # Realize a lazy backbone so self.parameters() is populated before configure_optimizers
        # (a lazy model — e.g. TimmModel — builds its backbone only on first forward / realize()).
        realize = getattr(self.model, "realize", None)
        if callable(realize):
            realize()
        # Build the metric set with the resolved count — only when not already built eagerly
        # in __init__ (the explicit-num_classes path); this covers the derived (omitted) path.
        if self.train_metrics is None:
            base = _build_classification_metrics(self.num_classes)
            self.train_metrics = base.clone(prefix="train/")
            self.val_metrics = base.clone(prefix="val/")
            self.test_metrics = base.clone(prefix="test/")

    def _apply_loader_knobs(self) -> None:
        """Inject batch_size / num_workers / shuffle into the deferred loader stubs.

        ``setdefault`` so an explicitly-configured loader keeps its own values.
        """
        persistent = bool(self.persist_workers and self.num_workers > 0)
        for loader, shuffle in ((self.train_loader, True), (self.val_loader, False), (self.test_loader, False)):
            if isinstance(loader, Fluid):
                loader.kwargs.setdefault("batch_size", self.batch_size)
                loader.kwargs.setdefault("num_workers", self.num_workers)
                loader.kwargs.setdefault("shuffle", shuffle)
                loader.kwargs.setdefault("persistent_workers", persistent)

    def _apply_lightning_knobs(self) -> None:
        """Build a flowable Trainer ``Class`` from the knobs when ``lightning`` is the default.

        The default ``self.lightning`` is a deferred ``LazyClass(L.Trainer)`` (protected from
        deep-flow); the base ``run()`` flows it with no runtime kwargs, which would leave a
        ``Lazy`` deferred. So materialise it into a concrete ``Class(L.Trainer, …)`` carrying
        the knobs. The ``logger`` list of ``!class:`` Fluids is pre-flowed (``flow()`` does not
        recurse into a list-valued kwarg). A caller-supplied ``lightning:`` ``Class`` is kept.
        """
        if isinstance(self.lightning, LazyClass):
            kwargs: dict[str, Any] = {"max_epochs": self.max_epochs, "accelerator": "auto", "devices": 1}
            kwargs.update(self.lightning.kwargs)
            if self.lightning_loggers:
                kwargs.setdefault("logger", [flow(x) if isinstance(x, Fluid) else x for x in self.lightning_loggers])
            self.lightning = Class(L.Trainer, **kwargs)
