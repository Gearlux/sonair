"""LightningClassificationTrainer: specialised image-classification trainer.

Subclass of :class:`~marainer.trainer.LightningTrainer` that bakes in all
classification-specific defaults so a minimal YAML only needs the dataset and
the model backbone — no adapter wiring for loss, collate, metrics, loaders, or
the Lightning Trainer, and no ``num_classes`` (it is derived from the dataset).

Baked-in defaults (all overridable via YAML ``@configurable`` attributes):

* ``loss_fn`` — :class:`torch.nn.CrossEntropyLoss` (override via the ``loss_fn``
  slot; any ``task="classification", role="loss"`` class is offered)
* ``train/val/test_loader`` — built from the ``batch_size`` / ``num_workers`` knobs,
  with :func:`~sonair.classification.classification_collate_fn`
* ``optimizer`` — ``AdamW(lr=1e-4, weight_decay=0.05)``
* ``lightning`` — ``pytorch_lightning.Trainer`` built from ``max_epochs`` + ``logger``
* Metrics — the ``metrics`` dict (default top-1 ``MulticlassAccuracy`` + macro
  ``MulticlassF1Score``), flowed with ``num_classes`` and cloned per split.

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

import inspect
from typing import Any, Iterable, List, Literal, Optional, Union, cast

import torch
import torch.nn as nn
from confluid import LazyClass, Mandatory, configurable, flow, output
from confluid.fluid import Fluid
from confluid.registry import resolve_class
from dataflux import Flux, LabelMap, iter_targets
from loggair import get_logger
from marainer.trainer import LightningTrainer, build_metric_collection, ensure_sample_dataset
from pytorch_lightning.loggers import Logger
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from tqdm import tqdm

from sonair.classification import classification_collate_fn

logger = get_logger(__name__)

#: How a sample's ``target`` is shaped — drives the encode / num_classes path.
_TargetKind = Literal["int", "str", "multilabel", "empty"]


def _loss_accepts_weight(loss: Any) -> bool:
    """True when ``loss``'s CONSTRUCTOR declares a ``weight`` parameter (balance-capability check).

    Works whether ``loss`` is still deferred (a confluid ``Fluid`` / ``!lazy:`` — resolved to its
    target class via ``confluid.registry.resolve_class``) or already a live ``nn.Module``. The ctor
    signature, not ``hasattr(loss, "weight")``, is authoritative: it tells us whether the weight can
    be supplied at construction (so a balanced ``weight`` tensor can be flowed in / set).
    """
    loss_cls = resolve_class(loss.target) if isinstance(loss, Fluid) else type(loss)
    if loss_cls is None:
        return False
    try:
        # signature(cls) gives the constructor's call signature (without `self`).
        return "weight" in inspect.signature(loss_cls).parameters
    except (ValueError, TypeError):  # pragma: no cover - C-extension ctor without a signature
        return False


def _target_to_int(value: Any) -> int:
    """Coerce a single scalar class id into a Python ``int`` (rejects bool)."""
    if isinstance(value, bool):
        raise TypeError(f"target {value!r} is a bool, not a class id")
    return int(value)


def detect_target_kind(source: Any) -> _TargetKind:
    """Classify a dataset's target by peeking the first non-``None`` target (cheap projection).

    Shared by the classification trainer and evaluator so both decide the encode / num_classes
    path identically. ``"int"`` (scalar class id), ``"str"`` (class name), ``"multilabel"`` (a
    sequence of names — not yet supported), or ``"empty"`` (no targets).
    """
    for target in iter_targets(flow(source)):
        if target is None:
            continue
        if isinstance(target, str):
            return "str"
        if isinstance(target, (list, tuple, set)):
            return "multilabel"
        return "int"
    return "empty"


def encode_flux(source: Any, label_map: LabelMap) -> Flux:
    """Wrap ``source`` in a :class:`~dataflux.Flux` applying ``label_map``'s encode op.

    The encoded ``target`` is a plain class-id ``int`` (``EncodeTargetOp`` is framework-agnostic);
    the DataLoader collate turns the per-batch list of ids into the target tensor. Exposes
    ``label_names`` on the returned Flux so marainer's ``_resolve_label_names`` persists
    ``class_names.json`` (the file the evaluator/predict path reloads via ``LabelMap.load``).
    """
    flux = Flux(source=flow(source), ops=[label_map.encode_op()])
    flux.label_names = label_map.label_names  # type: ignore[attr-defined]
    return flux


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
    * Metrics — the ``metrics`` dict of single-metric ``LazyClass`` templates
      (default top-1 ``MulticlassAccuracy`` + macro ``MulticlassF1Score``; add top-5
      etc. by extending the dict). :meth:`initialize` flows each with the resolved
      ``num_classes``, wraps them in a ``MetricCollection``, and clones it per split
      (``train/`` · ``val/`` · ``test/``). **sonair OWNS the num_classes resolution**
      (:meth:`initialize`): a concrete model's ``num_classes`` (authoritative head size) →
      explicit trainer arg → the ``label_map`` → a lazy model's pinned ``num_classes`` →
      ``dataflux.num_classes(train_set)`` (walks integer targets); it then flows the model
      + metrics with that count directly — no generic dimension-injection layer. String
      targets are auto-encoded via a ``dataflux.LabelMap`` and the loss is class-balanced
      (see ``label_map`` / ``balance_classes`` below).

    **Ergonomic flat-config knobs** — ``batch_size`` / ``num_workers`` / ``max_epochs`` /
    ``logger`` are constructor params, so Confluid broadcasting injects matching flat
    top-level YAML keys. ``batch_size`` / ``num_workers`` are baked into the deferred
    ``DataLoader`` stubs in ``__init__``; ``max_epochs`` / ``logger`` are forwarded to the
    base, which bakes them into ``LazyClass(L.Trainer, max_epochs=…, logger=…)``.

    Args:
        model: Classification model emitting ``[N, num_classes]`` logits (wire ``!lazy:`` so
            :meth:`initialize` flows it with the resolved num_classes at run time).
        train_set: Training dataset or Flux pipeline.
        num_classes: Output class count; omit to resolve it from the model config or
            ``dataflux.num_classes(train_set)`` in :meth:`initialize`.
        max_epochs: ``pytorch_lightning.Trainer(max_epochs=...)`` (forwarded to the base Trainer).
        batch_size: DataLoader batch size baked into the default loaders in ``__init__``.
        num_workers: DataLoader worker-process count baked into the default loaders in ``__init__``.
        logger: Optional list of Lightning loggers (``List``-typed so a flat ``logger: [...]``
            broadcasts in); forwarded to the base as ``L.Trainer(logger=[...])``.
        experiment_name: MLflow experiment name.
        run_name: MLflow run name. ``None`` (default) auto-generates a random slug like
            ``train_funny_rabbit`` (via the base; see :func:`marainer.run_names.generate_run_name`).
        label_map: Optional :class:`~dataflux.LabelMap` (name→id) pinning the class encoding. When
            given, ``num_classes`` is taken from it and string targets are encoded through it. When
            omitted and the dataset's targets are *strings*, :meth:`initialize` fits one from the
            train set (sklearn ``LabelEncoder``). ``None`` (default) + integer targets ⇒ no encoding.
        balance_classes: When ``True`` (default), :meth:`initialize` walks the (encoded) train
            targets and injects inverse-frequency class weights into ``loss_fn`` (when the loss
            accepts a ``weight`` tensor, e.g. ``CrossEntropyLoss``); the weights are exposed on
            ``self.class_weights``. Set ``False`` to leave the loss unweighted.
        metrics: Dict of single-metric ``LazyClass`` templates flowed per split in
            :meth:`initialize` (default top-1 accuracy + macro F1).
    """

    def __init__(
        self,
        model: Mandatory[nn.Module],
        train_set: Mandatory[Dataset[Any]],
        num_classes: Optional[int] = None,
        max_epochs: int = 10,
        batch_size: int = 32,
        num_workers: int = 0,
        logger: Optional[Union[Logger, Iterable[Logger], bool]] = None,
        experiment_name: str = "sonair_classification",
        run_name: Optional[str] = None,
        label_map: Optional[LabelMap] = None,
        balance_classes: bool = True,
        metrics: dict[str, Metric | LazyClass[Metric]] | MetricCollection = {
            "accuracy_top1": LazyClass[Metric](MulticlassAccuracy, top_k=1),
            "f1_macro": LazyClass[Metric](MulticlassF1Score, average="macro"),
        },
    ) -> None:
        super().__init__(
            model=model,
            train_set=train_set,
            experiment_name=experiment_name,
            run_name=run_name,
            logger=logger,
            max_epochs=max_epochs,
        )
        # Store config ONLY — no functional work. The loaders/optimizer are deferred
        # (LazyClass): the loaders bake in the ergonomic batch_size/num_workers knobs
        # here (cheap — LazyClass defers actual construction), and the base run() flows
        # them with the dataset (the optimizer with model params). max_epochs / logger
        # are forwarded to the base, which bakes them into LazyClass(L.Trainer, ...).
        # The model + metrics are flowed with num_classes in initialize() (run time).
        self.num_classes = int(num_classes) if num_classes is not None else None
        # Optional pinned name->int label map (or fitted from the train set in initialize() when the
        # targets are strings); `class_weights` is the inverse-frequency balanced-loss weight derived
        # from the walked targets. Both are exposed (read after initialize()).
        self.label_map = label_map
        self.balance_classes = balance_classes
        self.class_weights: Optional[torch.Tensor] = None
        self.max_epochs = max_epochs
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = LazyClass(torch.optim.AdamW, lr=1e-4, weight_decay=0.05)
        self.train_loader = LazyClass(
            DataLoader,
            collate_fn=classification_collate_fn,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers != 0,
            shuffle=True,
        )
        self.val_loader = LazyClass(
            DataLoader,
            collate_fn=classification_collate_fn,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers != 0,
            shuffle=False,
        )
        self.test_loader = LazyClass(
            DataLoader,
            collate_fn=classification_collate_fn,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers != 0,
            shuffle=False,
        )
        self.metrics = metrics

    def run(self) -> None:
        """Resolve num_classes + flow the model and metrics, then run the base loop."""
        self.initialize()
        super().run()

    @property
    @output
    def trained_model(self) -> nn.Module:
        """The trained classification model (a live ``nn.Module``) produced by :meth:`run`.

        This is the trainer's declared confluid ``@output`` (the I/O contract FluxStudio /
        navigaitor read via :func:`confluid.output_specs`): the post-run artifact is the
        trained model, wired downstream into an evaluator's ``model`` slot. It is the live
        ``self.model`` that :meth:`initialize` flowed (lazy backbone solidified) and the
        Lightning ``fit`` loop trained in place — NOT the ``model`` *input* (the ``!lazy:``
        architecture spec). Accessing it before :meth:`run` raises, because the model is
        still a deferred :class:`~confluid.fluid.Fluid`: it is a run OUTPUT, not a config knob.
        """
        if isinstance(self.model, Fluid):
            raise RuntimeError(
                "LightningClassificationTrainer.trained_model is only available after run(): "
                "the model is still a deferred Fluid. Wire this output downstream of a Run node."
            )
        return cast(nn.Module, self.model)

    def initialize(self) -> None:
        """Resolve the label encoding + ``num_classes``, then flow the model + metrics with it.

        sonair OWNS this (no generic dimension-injection layer). The work, in order:

        1. **Detect the target kind** (cheap — peeks one target): integer class ids, class-name
           *strings*, or a sequence of names (multi-label, not yet supported).
        2. **Label encoding** — when the targets are strings: use the pinned ``label_map`` if given,
           else fit a :class:`~dataflux.LabelMap` from the train targets (sklearn ``LabelEncoder``,
           shown with a progress bar). The wired ``train``/``val``/``test`` sets are then wrapped in
           a :class:`~dataflux.Flux` carrying the map's ``encode_op`` so targets become ints, and the
           map's ``label_names`` is exposed on the train set so marainer persists ``class_names.json``
           (which the evaluator/predict path reloads via ``LabelMap.load``).
        3. **Resolve ``num_classes``** — a **concrete (non-``Fluid``) model is authoritative**
           (``getattr(model, "num_classes")``): its classifier head is already sized, so the count
           MUST come from it, not from a dataset walk that could mis-size the metrics. Otherwise:
           explicit trainer arg → the ``label_map`` → the lazy model's pinned ``num_classes`` →
           derived from the data (``max(class_id)+1``).
        4. **Balanced loss** — when ``balance_classes`` and the loss accepts a ``weight`` tensor,
           inject inverse-frequency class weights computed from the walked targets (exposed on
           ``self.class_weights``).
        5. Flow the model with the resolved count (``flow`` auto-solidifies a lazy ``TimmModel``
           backbone so its parameters exist before ``configure_optimizers``) and flow each
           single-metric template into per-split ``MetricCollection``s.

        Called explicitly from :meth:`run` (NOT the auto-fired confluid ``solidify`` hook) so the
        dataset walk + model build happen at run time. Idempotent: re-running reuses the resolved
        count, fitted map, and built metrics.
        """
        # Normalize the wired sources to a Dataset[Sample] up front (marainer helper) so EVERYTHING
        # below — target detection, label fitting, encoding, the loaders, collate — operates on
        # guaranteed Sample items and never has to re-check the source shape.
        self.train_set = ensure_sample_dataset(flow(self.train_set))
        if self.val_set is not None:
            self.val_set = ensure_sample_dataset(flow(self.val_set))
        if self.test_set is not None:
            self.test_set = ensure_sample_dataset(flow(self.test_set))

        kind = detect_target_kind(self.train_set)
        if kind == "multilabel":
            raise NotImplementedError(
                "LightningClassificationTrainer: multi-label string targets are not yet supported "
                "(needs multi-hot encoding + a BCE-style loss + collate changes). See TASKS.md."
            )

        # Resolve num_classes from everything that does NOT need a data walk (a concrete model's
        # head, an explicit arg, a given label_map, a lazy model's pinned kwarg).
        nc = self._num_classes_without_data()

        # The train targets are walked AT MOST ONCE, then reused for every consumer below — fitting
        # the label encoder, deriving num_classes from the data, and computing balance weights.
        want_weights = self.balance_classes and _loss_accepts_weight(self.loss_fn)
        need_fit = kind == "str" and self.label_map is None
        raw_targets: Optional[List[Any]] = None
        if need_fit or want_weights or nc is None:
            raw_targets = self._walk_targets(self.train_set, "Reading targets")

        # --- Label encoding (string targets). Fit from the RAW (pre-encode) targets, then wrap. ---
        if need_fit:
            assert raw_targets is not None
            self.label_map = LabelMap.fit(raw_targets)
            logger.info(f"Fitted LabelMap with {self.label_map.num_classes} classes from train targets.")
        if kind == "str" and self.label_map is not None:
            self.train_set = encode_flux(self.train_set, self.label_map)
            if self.val_set is not None:
                self.val_set = encode_flux(self.val_set, self.label_map)
            if self.test_set is not None:
                self.test_set = encode_flux(self.test_set, self.label_map)

        # --- Finalize num_classes from the walked targets when no non-data source supplied it. ---
        if nc is None:
            if self.label_map is not None:
                nc = self.label_map.num_classes
            else:
                assert raw_targets is not None
                highest = max((_target_to_int(t) for t in raw_targets if t is not None), default=-1)
                nc = highest + 1 if highest >= 0 else None
        if not nc:
            raise RuntimeError(
                "LightningClassificationTrainer: could not determine num_classes — set it on the "
                "trainer or the model, pass a label_map, or wire a dataset whose targets yield a count."
            )
        self.num_classes = int(nc)

        # --- Balanced loss: inverse-frequency weights from the SAME walked targets (no second walk).
        # _apply_class_weights warns + no-ops for a non-weight-capable loss; when capable, want_weights
        # was True above so raw_targets is already populated (no second walk). ---
        if self.balance_classes and self.class_weights is None:
            self._apply_class_weights(self.num_classes, kind, raw_targets)

        # Flow the model with the resolved count (flow auto-solidifies a lazy backbone).
        self.model = flow(self.model, num_classes=self.num_classes)
        # Build the per-split metric collections from the shared helper (handles a dict of templates
        # vs a ready-made MetricCollection) and clone the base per split.
        if self.train_metrics is None:
            base = build_metric_collection(self.metrics, num_classes=self.num_classes)
            self.train_metrics = base.clone(prefix="train/")
            self.val_metrics = base.clone(prefix="val/")
            self.test_metrics = base.clone(prefix="test/")

    # ------------------------------------------------------------------ helpers

    def _walk_targets(self, source: Any, desc: str) -> List[Any]:
        """Walk every target with a progress bar (target-only projection skips input construction)."""
        src = flow(source)
        total = len(src) if hasattr(src, "__len__") else None
        return list(tqdm(iter_targets(src), total=total, desc=f"sonair: {desc}", unit="sample"))

    def _num_classes_without_data(self) -> Optional[int]:
        """Resolve the class count from sources that need NO data walk, else ``None``.

        A concrete (non-``Fluid``) model's classifier head is authoritative; then an explicit trainer
        arg, a given ``label_map``, or a lazy model's pinned ``num_classes``. Returns ``None`` when the
        count can only come from walking the targets (left to the caller, which walks once).
        """
        if not isinstance(self.model, Fluid):
            model_nc = getattr(self.model, "num_classes", None)
            if model_nc:
                if self.num_classes is not None and int(self.num_classes) != int(model_nc):
                    logger.warning(
                        f"Trainer num_classes={self.num_classes} != concrete model num_classes={model_nc}; "
                        "using the model's (its classifier head is already sized)."
                    )
                return int(model_nc)
        if self.num_classes is not None:
            return int(self.num_classes)
        if self.label_map is not None:
            return self.label_map.num_classes
        if isinstance(self.model, Fluid):
            pinned = self.model.kwargs.get("num_classes")
            if pinned:
                return int(pinned)
        return None

    def _apply_class_weights(self, num_classes: int, kind: _TargetKind, raw_targets: Optional[List[Any]]) -> None:
        """Compute inverse-frequency class weights from the ALREADY-walked ``raw_targets`` and inject them.

        The loss is balance-capable only if its constructor takes a ``weight`` (see
        :func:`_loss_accepts_weight`); otherwise we warn and leave it unweighted. When capable, a
        deferred (``!lazy:``) loss gets the weight FLOWED IN at construction, and a live loss gets its
        ``weight`` buffer set. Never walks the data itself — ``raw_targets`` is the single shared walk.
        """
        if not _loss_accepts_weight(self.loss_fn):
            if isinstance(self.loss_fn, Fluid):
                name: Any = getattr(self.loss_fn, "target", "loss")
            else:
                name = type(self.loss_fn).__name__
            logger.warning(
                f"balance_classes=True but the configured loss ({name}) has no 'weight' constructor "
                "argument — it is not balance-capable; training proceeds with an unweighted loss. "
                "Use a weight-accepting loss (e.g. CrossEntropyLoss / NLLLoss / BCEWithLogitsLoss) to balance."
            )
            self.class_weights = None
            return

        assert raw_targets is not None  # capable ⇒ initialize() walked the targets already
        # raw_targets are RAW (pre-encode): strings for the str path (encode via the map), ints otherwise.
        if kind == "str":
            assert self.label_map is not None
            encoded = [self.label_map.mapping[str(t)] for t in raw_targets if t is not None]
        else:
            encoded = [_target_to_int(t) for t in raw_targets if t is not None]

        counts = torch.zeros(num_classes, dtype=torch.float64)
        for c in encoded:
            if 0 <= c < num_classes:
                counts[c] += 1
        total = float(counts.sum())
        if total <= 0:
            return
        # Inverse frequency, normalised so present classes average to ~1. Absent classes → 0.
        weights = torch.where(counts > 0, total / (num_classes * counts), torch.zeros_like(counts))
        self.class_weights = weights.to(torch.float32)
        if isinstance(self.loss_fn, Fluid):
            # Deferred loss — flow the weight directly into its constructor.
            self.loss_fn = flow(self.loss_fn, weight=self.class_weights)
        else:
            # Live loss — `weight` is a registered buffer on weighted losses, so assignment routes
            # into `_buffers` and Lightning moves it to device.
            self.loss_fn.weight = self.class_weights
        logger.info(f"Balanced loss: injected inverse-frequency class weights for {num_classes} classes.")
