"""``TorchClassifier`` — the same classification run as a PLAIN PyTorch training loop.

The eighth backend, and the one with nothing underneath it. Every other backend hands the epoch
to somebody: Lightning to a ``Trainer``, Keras to ``fit()``, fastai to a ``Learner``,
transformers to its own ``Trainer``. This one writes the loop out — ``zero_grad`` /
``backward`` / ``step``, an explicit device, an explicit ``model.train()`` / ``model.eval()``,
an explicit ``torch.no_grad()`` on the eval pass.

**Why it earns its place in a project whose point is that the config is the program.** It is the
control experiment. The claim every other backend makes is "swap the runnable, keep the config";
this one removes the training *framework* entirely and the config still does not change — same
dataset, same ``TimmModel``, same ``torch.nn`` losses, same ``torchmetrics``, same artifacts. If
a knob only worked because Lightning implemented it, this backend is where that shows up.

It is also the smallest correct answer to "how do I train this on plain torch?", which is a real
question: ``pip install 'sonair[torch]'`` brings torch, timm and torchmetrics and NOT
``pytorch_lightning``.

**What it shares with the Lightning backend, and what it does not.** The model
(:class:`~sonair.timm_model.TimmModel`), the losses and metrics
(:mod:`sonair.torch_classification`, ``framework="torch"``), the batch shape
(``[N, 3, H, W]`` via ``collate_records``) and the run artifacts are the SAME objects — a
``!scope:framework=torch`` block overrides the runnable and nothing else. What differs is the
loop and what a loop implies: there is no ``Trainer``, so there is no ``logger:`` slot of
framework loggers (tracking goes through marainer ``trackers:``, which every backend has), no
callback protocol, and no distributed strategy. Multi-GPU is deliberately out of scope — that is
precisely what Lightning is for, and pretending otherwise here would be a worse ``Trainer``.

The label pipeline is DUPLICATED from the other backends rather than shared, deliberately and
temporarily: an engine-neutral base is deferred until the detection projects have been through
the same treatment, so it can be designed against classification AND detection instead of
against classification six times. See ``TASKS.md``.

Minimal flat YAML — the Lightning block with one key changed::

    runnable: !class:sonair.TorchClassifier
      model: !lazy:sonair.TimmModel { model_name: efficientvit_b0 }
    train_set: !class:recordstream.Stream { source: ..., ops: !ref:preprocess }
    batch_size: 32
    max_epochs: 3

Run it with ``sonair train cfg.yaml --framework torch``.
"""

import inspect
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Union, cast

import numpy as np
import torch
import torch.nn as nn
from confluid import Fluid, Lazy, LazyClass, Mandatory, configurable, flow, output
from confluid.registry import resolve_class
from loggair import get_logger
from marainer.artifacts import load_class_names, save_class_names, write_metrics_sidecar
from marainer.metric_slots import merge_metric_specs
from marainer.run_names import generate_run_name, run_prefix
from marainer.torch import Device, build_metric_collection, resolve_device, scalar_metrics
from marainer.tracking.base import Tracker, as_tracker_list
from marainer.tracking.provenance import capture_log_file
from marainer.tracking.render import render_matrix_as_figure, render_matrix_as_table
from recordstream import (
    ClassificationOutput,
    LabelMap,
    PredictionsSink,
    RecordSource,
    batch_metadata,
    batch_tensor,
    class_names,
    classification_output,
    collate_records,
    ensure_record_dataset,
    first_value,
    inverse_frequency_weights,
    is_class_id,
    iter_key,
    multi_hot,
)
from recordstream.ops.image import confusion_matrices_payload
from recordstream.runnable import ProgressReporting, TorchRunner, entrypoint, run_entrypoint
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MultilabelAccuracy, MultilabelF1Score

logger = get_logger(__name__)

#: Which entry point :meth:`TorchClassifier.run` dispatches to — the same vocabulary every other
#: backend uses, because the verb is a property of the TASK, not of the engine.
ClassifierTask = Literal["fit", "evaluate", "test", "predict"]

#: What a metrics slot admits: a ``{name: template}`` mapping, a ready-made collection, or
#: ``None`` for "not chosen". Declared here rather than imported from the Lightning backend for
#: the reason this whole module exists — that import would pull ``pytorch_lightning`` into a
#: deliberately framework-free one (``test_it_needs_no_training_framework`` scans for exactly it).
MetricsSpec = Optional[Union[Dict[str, Union[Metric, LazyClass[Metric]]], MetricCollection]]

#: Re-exported so ``TorchClassifier``'s ``device`` knob can be read from this module the way
#: every other name in its signature can. It is marainer's (``marainer.torch.device``): a
#: ``Trainer`` picks an accelerator for you, a hand-written loop has to pick one itself, and
#: every such loop picks identically — so the choice is a torch-platform helper, not this
#: backend's private function.
__all__ = ["ClassifierTask", "Device", "TorchClassifier"]


@configurable(task="classification", role="trainer", framework="torch")
class TorchClassifier(TorchRunner, ProgressReporting):
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
        num_workers: DataLoader worker processes. Persistent workers follow from this, exactly
            as in the Lightning backend — there is no separate knob.
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
            (``marainer[plot]``) a cost of every eval epoch, where the table costs nothing.
        device: Where to run — see :data:`Device`.
        checkpoint_dir: Directory for ``model.pt`` (a plain ``state_dict``) and the run's
            ``class_names.json``. ``None`` writes no weights, which is what a smoke run wants.
        grad_clip: Clip gradients to this global norm before each ``step()``. ``None`` disables.

    Post-construction slots (YAML / broadcasting): ``val_set``, ``optimizer``,
    ``train/val/test_loader``, ``loss_fn``, ``predictions_sink``, ``metrics_sidecar_path``,
    ``trackers``.
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
        trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = None,
    ) -> None:
        super().__init__()
        # `nn.Module`, not `Lazy[nn.Module]`: the DEFERRED-ness is declared once, on the ctor
        # param above, which is what `lazy_param_names` / `to_pydantic` read. Restating it here
        # would make the attribute a union no method could call through.
        self.model: nn.Module = cast(nn.Module, model)
        self.train_set: Optional[Lazy[RecordSource]] = train_set
        self.val_set: Optional[Lazy[RecordSource]] = None
        self.test_set: Optional[Lazy[RecordSource]] = test_set
        self.task: ClassifierTask = task
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.experiment_name = experiment_name
        self.run_name = run_name if run_name is not None else generate_run_name(prefix=run_prefix(task))
        self.input_key = str(input_key)
        self.target_key = str(target_key)
        self.label_map = label_map
        self.balance_classes = balance_classes
        self.class_weights: Optional[torch.Tensor] = None
        self.device_choice: Device = device
        self.checkpoint_dir = checkpoint_dir
        self.grad_clip = grad_clip
        # Deferred runtime slots, built as `LazyClass` so a config can override any DataLoader
        # knob (`pin_memory`, `drop_last`, a sampler) without this class enumerating them — the
        # same construction the Lightning backend uses, and the reason `collate_records` is
        # baked in here rather than passed at the call site.
        loader_kw: Dict[str, Any] = dict(
            collate_fn=collate_records,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers != 0,
        )
        self.train_loader: Lazy[DataLoader[Any]] = LazyClass(DataLoader, shuffle=True, **loader_kw)
        self.val_loader: Lazy[DataLoader[Any]] = LazyClass(DataLoader, shuffle=False, **loader_kw)
        self.test_loader: Lazy[DataLoader[Any]] = LazyClass(DataLoader, shuffle=False, **loader_kw)
        self.optimizer: Lazy[Optimizer] = LazyClass(torch.optim.AdamW, lr=1e-4, weight_decay=0.05)
        # Sentinels: the right loss and metrics depend on whether the targets are single- or
        # multi-label, which is only known once `initialize()` has peeked the data. A live
        # default here could not be told apart from a user's explicit choice.
        self.loss_fn: Optional[Lazy[nn.Module]] = None
        self.metrics: Any = metrics
        self.multilabel: bool = False
        # Dual-purpose the way `model` is: that split's metrics SPEC until `initialize()` builds
        # it, the live prefixed collection after. `_metrics_built` tells the two states apart —
        # `is None` cannot, once a slot may legitimately arrive carrying a spec.
        self.train_metrics: Any = train_metrics
        self.val_metrics: Any = val_metrics
        self.test_metrics: Any = test_metrics
        self._metrics_built: bool = False
        self.plot_metrics = plot_metrics
        self.predictions_sink: Optional[Lazy[PredictionsSink]] = None
        self.metrics_sidecar_path: Optional[str] = None
        # Stored VERBATIM (lazy-init mandate: the constructor does no work). It may hold a
        # single tracker, a list, or deferred `Fluid` markers; `_live_trackers()` normalizes
        # and flows it at run time.
        self.trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = trackers
        self._class_names: Optional[List[str]] = None
        self._history: Dict[str, List[float]] = {}
        self._computed_metrics: Dict[str, float] = {}
        self._device: torch.device = torch.device("cpu")

    @property
    def __needs_autograd__(self) -> bool:  # type: ignore[override]  # dynamic per-task override
        """Only ``fit`` needs autograd; evaluate / test / predict are inference-only."""
        return self.task == "fit"

    # ---- run dispatch ------------------------------------------------------- #
    def run(self) -> None:
        self.initialize()
        # The `@entrypoint` markers below ARE the dispatch table (recordstream docs/runnable.md).
        run_entrypoint(self, self.task)

    # ---- setup -------------------------------------------------------------- #
    def initialize(self) -> None:
        """Resolve labels and ``num_classes``, build the model, and place everything on a device."""
        self._device = resolve_device(self.device_choice)
        logger.info(f"sonair torch backend on {self._device}.")
        if self.task == "fit":
            self._initialize_fit()
        else:
            self._initialize_eval()
        if not isinstance(self.model, nn.Module):
            raise TypeError(
                f"TorchClassifier.model must be a torch.nn.Module, got {type(self.model).__name__}. "
                "Wire `!lazy:sonair.TimmModel` (or any nn.Module) into the model slot."
            )
        # The model is BUILT by the `flow()` above, whether it arrived deferred or live —
        # confluid fires `solidify()` on its idempotency pass-through too (2026-08-02), so a
        # config wiring `!class:sonair.TimmModel()` (or a model handed in from Python) no longer
        # reaches the optimizer with an unbuilt backbone and NO parameters. That used to need a
        # `getattr(self.model, "solidify", None)` dance here, i.e. this backend re-implementing
        # confluid's own hook; the fix belonged in `flow()`, which had always promised it.
        self.model.to(self._device)
        # The loss moves too, and it is not decoration: a balanced `CrossEntropyLoss` holds its
        # class weights in a BUFFER, and a CPU buffer against CUDA logits raises mid-batch.
        loss = flow(self.loss_fn)
        if isinstance(loss, nn.Module):
            self.loss_fn = loss.to(self._device)

    def _initialize_fit(self) -> None:
        """Fit the label encoding, resolve ``num_classes``, then build the model and metrics.

        Deliberately the same sequence as every other backend's method of this name — none of it
        is engine-specific, and it is duplicated rather than shared until the base can be
        designed against detection too (see the module docstring).
        """
        self.train_set = ensure_record_dataset(self.train_set)
        if self.val_set is not None:
            self.val_set = ensure_record_dataset(self.val_set)
        if self.test_set is not None:
            self.test_set = ensure_record_dataset(self.test_set)

        # `iter_key` unwraps a MultiLabel to its list of values, so a sequence here IS a
        # multi-label target — the item type, not a guess about what a list might mean.
        first = first_value(self.train_set, self.target_key)
        self.multilabel = isinstance(first, (list, tuple, set))
        probe = next(iter(first), None) if self.multilabel else first
        targets_are_names = probe is not None and not is_class_id(probe)
        # BEFORE the balancing probe: `accepts_class_weights()` reads `loss_fn`, and an
        # unresolved `None` would answer "cannot be balanced" and skip the target walk.
        self._resolve_task_defaults()

        if targets_are_names and self.label_map is None:
            self.label_map = LabelMap.fit(iter_key(self.train_set, self.target_key))
            logger.info(f"Fitted LabelMap with {self.label_map.num_classes} classes from train targets.")
        if targets_are_names and self.label_map is not None:
            self.train_set = self.label_map.encode(self.train_set)
            if self.val_set is not None:
                self.val_set = self.label_map.encode(self.val_set)
            if self.test_set is not None:
                self.test_set = self.label_map.encode(self.test_set)

        want_weights = self.balance_classes and self._accepts_class_weights()
        raw_targets: Optional[List[Any]] = None
        if want_weights or self.num_classes is None:
            raw_targets = list(iter_key(self.train_set, self.target_key))

        if self.num_classes is None:
            if self.label_map is not None:
                self.num_classes = self.label_map.num_classes
            else:
                assert raw_targets is not None
                # An empty map still maps ids through — that is the `to_ids` contract.
                ids = [i for t in raw_targets for i in LabelMap().to_ids(t)]
                self.num_classes = max(ids) + 1 if ids else None
        if not self.num_classes:
            raise RuntimeError("could not determine num_classes — set it, or wire a labelled dataset.")
        self._class_names = class_names(self.train_set, self.val_set, self.test_set)

        if want_weights and raw_targets is not None:
            self._balance_loss(self.num_classes, raw_targets)

        self.model = flow(self.model, num_classes=self.num_classes)
        if not self._metrics_built:
            self.train_metrics = self._build_split_metrics(self.train_metrics, "train/")
            self.val_metrics = self._build_split_metrics(self.val_metrics, "val/")
            self.test_metrics = self._build_split_metrics(self.test_metrics, "test/")
            self._metrics_built = True

    def _initialize_eval(self) -> None:
        """Model-driven setup: the trained head's width is authoritative, not the eval split's.

        An eval dataset may be a subset that never shows every class, so deriving the count from
        it would mis-size the metrics against the head that produced the logits.
        """
        self.model = flow(self.model)
        nc = getattr(self.model, "num_classes", None)
        if not nc:
            raise RuntimeError("could not determine num_classes from the model (it must expose `num_classes`).")
        self.num_classes = int(nc)
        if self.test_set is not None:
            dataset = ensure_record_dataset(self.test_set)
            first = first_value(dataset, self.target_key)
            self.multilabel = isinstance(first, (list, tuple, set))
            probe = next(iter(first), None) if self.multilabel else first
            if probe is not None and not is_class_id(probe):
                label_map = self.label_map or load_class_names(getattr(self.model, "checkpoint_path", None))
                if label_map is None:
                    raise RuntimeError(
                        "eval dataset has string targets but no label_map and no class_names.json found."
                    )
                self.label_map = label_map
                dataset = label_map.encode(dataset)
            self.test_set = dataset
            self._class_names = class_names(dataset)
        self._resolve_task_defaults()
        if not self._metrics_built:
            self.test_metrics = self._build_split_metrics(self.test_metrics, "test/")
            self._metrics_built = True

    def _resolve_task_defaults(self) -> None:
        """Fill the ``loss_fn`` / ``metrics`` sentinels once the target shape is known.

        Identical choices to the Lightning backend, and that is the point: a config moved between
        the two must not silently change its loss. Idempotent, so an explicitly wired value is
        never replaced.
        """
        if self.loss_fn is None:
            # Multi-label needs per-class independent sigmoids; single-label a softmax.
            self.loss_fn = nn.BCEWithLogitsLoss() if self.multilabel else nn.CrossEntropyLoss()
            logger.debug(f"loss_fn defaulted to {type(self.loss_fn).__name__} (multilabel={self.multilabel}).")
        if self.metrics is None:
            self.metrics = (
                {
                    "accuracy": LazyClass[Metric](MultilabelAccuracy),
                    "f1_macro": LazyClass[Metric](MultilabelF1Score, average="macro"),
                }
                if self.multilabel
                else {
                    "accuracy_top1": LazyClass[Metric](MulticlassAccuracy, top_k=1),
                    "f1_macro": LazyClass[Metric](MulticlassF1Score, average="macro"),
                }
            )

    def _build_split_metrics(self, spec: Any, prefix: str) -> MetricCollection:
        """One split's live, prefixed collection — its own spec merged over the shared one.

        The combination rule is :func:`~marainer.metric_slots.merge_metric_specs`, shared with
        every other backend so a config moving between them cannot change shape; what is torch's
        alone is the rest of this line — sizing the templates and prefixing the collection.
        """
        return build_metric_collection(merge_metric_specs(self.metrics, spec), **self._metric_dims()).clone(
            prefix=prefix
        )

    def _metric_dims(self) -> Dict[str, int]:
        """The sizing kwarg metric templates are flowed with.

        torchmetrics sizes multi-label metrics by ``num_labels`` and multi-class ones by
        ``num_classes`` — same number, different keyword.
        """
        assert self.num_classes is not None
        return {"num_labels": self.num_classes} if self.multilabel else {"num_classes": self.num_classes}

    # ---- class balancing: the STATISTIC is recordstream's, the INJECTION is this backend's -- #
    def _accepts_class_weights(self) -> bool:
        """Whether the wired loss can consume per-class weights.

        Answered for torch: a ``torch.nn`` loss takes a ``weight`` TENSOR in its constructor, so
        the question is whether the signature declares one. Works on a live loss OR a deferred
        :class:`~confluid.fluid.Fluid`, which matters — this is asked BEFORE the loss is built,
        to decide whether walking the whole target stream is worth it at all.
        """
        loss_cls = resolve_class(self.loss_fn.target) if isinstance(self.loss_fn, Fluid) else type(self.loss_fn)
        if loss_cls is None:
            return False
        try:
            return "weight" in inspect.signature(loss_cls).parameters
        except (ValueError, TypeError):  # pragma: no cover - a C-extension ctor without a signature
            return False

    def _balance_loss(self, num_classes: int, raw_targets: List[Any]) -> None:
        """Inject inverse-frequency class weights into the loss.

        The WEIGHTS are a statistic over the labels, so recordstream computes them (and returns
        numpy); what a framework does with the vector is this backend's business — for torch,
        a tensor into the loss's ``weight``.

        The two-branch injection is not a stylistic choice. A deferred loss must be FLOWED with
        ``weight=`` because a ``Fluid`` cannot be mutated in place; a LIVE one must have the
        attribute set, because ``flow()`` on a live object returns that same object with the
        runtime kwarg DROPPED — silently unweighted training, which is exactly the failure this
        method exists to prevent (measured: a live ``CrossEntropyLoss`` came back with
        ``weight=None``).
        """
        weights = inverse_frequency_weights(raw_targets, num_classes, self.label_map)
        if weights is None:
            logger.warning("Class balancing requested but no class ids were observed; training unweighted.")
            return
        tensor = torch.as_tensor(np.asarray(weights), dtype=torch.float32)
        if isinstance(self.loss_fn, Fluid):
            self.loss_fn = flow(self.loss_fn, weight=tensor)
        else:
            cast(nn.Module, self.loss_fn).weight = tensor
        self.class_weights = tensor
        logger.info(f"Balanced the loss with inverse-frequency class weights over {num_classes} classes.")

    # ---- the batch boundary -------------------------------------------------- #
    def _targets(self, batch: Any) -> torch.Tensor:
        """The target tensor this task's loss expects — the ONLY task-shaped choice left.

        Both branches are recordstream calls; what is classification-specific is WHICH one and
        the dtype named for it. ``int64`` explicitly, because a dataset yielding int32 label
        tensors is legal and ``CrossEntropyLoss`` rejects it with *"expected target dtype to be
        Long or Byte, but got Int"*.
        """
        if self.multilabel:
            if not self.num_classes:
                raise RuntimeError("multi-label targets need num_classes — call initialize() first.")
            return torch.as_tensor(multi_hot(batch, self.target_key, self.num_classes)).to(self._device)
        return batch_tensor(batch, self.target_key, self._device, dtype=torch.int64)

    def _loaders(self) -> Dict[str, Optional[DataLoader[Any]]]:
        """The deferred loader slots, flowed with the datasets this run resolved."""
        return {
            "train": flow(self.train_loader, dataset=self.train_set) if self.train_set is not None else None,
            "val": flow(self.val_loader, dataset=self.val_set) if self.val_set is not None else None,
            "test": flow(self.test_loader, dataset=self.test_set) if self.test_set is not None else None,
        }

    # ---- the loop ------------------------------------------------------------ #
    def _train_epoch(self, loader: DataLoader[Any], optimizer: Optimizer) -> float:
        """One training pass, returning the mean loss. The whole reason this backend exists."""
        model, loss_fn = cast(nn.Module, self.model), cast(nn.Module, flow(self.loss_fn))
        model.train()
        total, batches = 0.0, 0
        for batch in loader:
            x = batch_tensor(batch, self.input_key, self._device)
            y = self._targets(batch)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
            optimizer.step()
            total += float(loss.detach())
            batches += 1
            if self.train_metrics is not None:
                self.train_metrics.update(logits.detach(), y)
        return total / batches if batches else float("nan")

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader[Any], metrics: Optional[MetricCollection]) -> float:
        """One evaluation pass, returning the mean loss. ``no_grad`` because nothing here trains."""
        model, loss_fn = cast(nn.Module, self.model), cast(nn.Module, flow(self.loss_fn))
        model.eval()
        total, batches = 0.0, 0
        for batch in loader:
            x = batch_tensor(batch, self.input_key, self._device)
            y = self._targets(batch)
            logits = model(x)
            total += float(loss_fn(logits, y))
            batches += 1
            if metrics is not None:
                metrics.update(logits, y)
        return total / batches if batches else float("nan")

    def _epoch_metrics(
        self,
        collection: Optional[MetricCollection],
        trackers: Sequence[Tracker] = (),
        step: Optional[int] = None,
    ) -> Dict[str, float]:
        """Compute and RESET a split's metrics — the bookkeeping a ``Trainer`` hides.

        Only the SCALARS are RETURNED: they go on to `Tracker.log_metrics`, which is a
        `{name: float}` channel in every backend. A non-scalar metric (an N×N confusion
        matrix) has no place there and would otherwise vanish silently, so it takes the
        other route — rendered into the log and pushed to each tracker's `log_text`, which
        is the channel that exists for exactly this. recordstream decides which computed
        values ARE confusion matrices (square 2-D, by SHAPE not by metric name), marainer
        renders the table.
        """
        if collection is None:
            return {}
        computed = collection.compute()
        for payload in confusion_matrices_payload(computed, self._class_names or None):
            table = render_matrix_as_table(payload["name"], payload["counts"], payload["class_names"])
            logger.info(table)
            # Rendered ONCE per report, not once per tracker: a figure is real work, and every
            # tracker can be handed the same one.
            figure = (
                render_matrix_as_figure(payload["name"], payload["counts"], payload["class_names"])
                if self.plot_metrics and trackers
                else None
            )
            for tracker in trackers:
                tracker.log_text(payload["name"], table, step=step)
                if figure is not None:
                    tracker.log_figure(payload["name"], figure, step=step)
        values = scalar_metrics(computed)
        collection.reset()
        return values

    # ---- the four entry points ---------------------------------------------- #
    @entrypoint("fit", role="trainer", primary=True)
    def fit(self) -> None:
        """Epochs, in the open: forward, backward, step, validate, log, checkpoint."""
        logger.info(f"sonair torch training: {self.experiment_name} ({self.run_name})")
        if self.train_set is None:
            raise ValueError("TorchClassifier.fit() requires a 'train_set'.")
        loaders = self._loaders()
        train_loader = cast(DataLoader[Any], loaders["train"])
        optimizer = cast(Optimizer, flow(self.optimizer, params=cast(nn.Module, self.model).parameters()))
        trackers = self._live_trackers()
        manifest = self._log_run_context(trackers)
        self._move_metrics()

        for epoch in range(self.max_epochs):
            logs: Dict[str, float] = {"train/loss": self._train_epoch(train_loader, optimizer)}
            logs.update(self._epoch_metrics(self.train_metrics, trackers, epoch))
            if loaders["val"] is not None:
                logs["val/loss"] = self._eval_epoch(cast(DataLoader[Any], loaders["val"]), self.val_metrics)
                logs.update(self._epoch_metrics(self.val_metrics, trackers, epoch))
            for tracker in trackers:
                tracker.log_metrics(logs, step=epoch)
            for key, value in logs.items():
                self._history.setdefault(key, []).append(value)
            logger.info(f"epoch {epoch + 1}/{self.max_epochs}: " + ", ".join(f"{k}={v:.4f}" for k, v in logs.items()))
            self._computed_metrics = dict(logs)

        logger.info("Training complete.")
        self._save_checkpoint()
        save_class_names(self._class_names, self.checkpoint_dir, trackers)
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)
        if loaders["test"] is not None:
            self._computed_metrics.update(self._score(cast(DataLoader[Any], loaders["test"])))
        # Uploads the log file AND stamps the manifest's terminal status. Not optional once a
        # manifest exists: left unfinished it stays `running`, and marainer's atexit hook then
        # calls a successful run `interrupted`.
        capture_log_file(trackers, manifest=manifest, status="success")

    @entrypoint("evaluate", role="evaluator")
    def evaluate(self) -> None:
        self.test()

    @entrypoint("test", role="evaluator", primary=True)
    def test(self) -> None:
        if self.test_set is None:
            raise ValueError("TorchClassifier.test() requires a 'test_set'.")
        self._move_metrics()
        self._computed_metrics = self._score(cast(DataLoader[Any], self._loaders()["test"]))
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)

    def _score(self, loader: DataLoader[Any]) -> Dict[str, float]:
        """Evaluate over ``loader`` and report — shared by ``test`` and the post-fit test pass."""
        trackers = self._live_trackers()
        results = {"test/loss": self._eval_epoch(loader, self.test_metrics)}
        results.update(self._epoch_metrics(self.test_metrics, trackers))
        for tracker in trackers:
            tracker.log_metrics(results)
        logger.info(f"Evaluation complete: {results}")
        return results

    @entrypoint("predict", role="predictor", primary=True)
    @torch.no_grad()
    def predict(self) -> None:
        """Stream per-record predictions to the sink, pairing each batch with its metadata."""
        logger.info(f"sonair torch predicting: {self.experiment_name} ({self.run_name})")
        if self.test_set is None:
            raise ValueError("TorchClassifier.predict() requires a 'test_set'.")
        model = cast(nn.Module, self.model)
        model.eval()
        sink = flow(self.predictions_sink)
        loader = cast(DataLoader[Any], self._loaders()["test"])
        for batch in loader:
            out = classification_output(model(batch_tensor(batch, self.input_key, self._device)))
            if sink is not None:
                self._write_predictions(out, batch, cast(PredictionsSink, sink))
        if sink is not None and hasattr(sink, "close"):
            sink.close()
        logger.info("Prediction complete.")

    # ---- predictions -------------------------------------------------------- #
    def _write_predictions(self, out: ClassificationOutput[torch.Tensor], batch: Any, sink: PredictionsSink) -> None:
        """One prediction per record, sliced from the batched output the model emits.

        The model returns ``probs`` ``[N, C]`` and ``class_idx`` ``[N]`` while the sink contract
        is per-record, so handing the batch over whole would write ONE entry for N records.
        """
        per_record_meta = batch_metadata(batch, exclude=(self.input_key, self.target_key))
        rows = len(out["class_idx"])
        for i in range(rows):
            item: ClassificationOutput[torch.Tensor] = {
                "logits": out["logits"][i],
                "probs": out["probs"][i],
                "class_idx": out["class_idx"][i],
            }
            meta = per_record_meta[i] if per_record_meta and i < len(per_record_meta) else {}
            sink.write(item, meta)

    # ---- artifacts + tracking ----------------------------------------------- #
    def _save_checkpoint(self) -> None:
        """A plain ``state_dict`` — no framework format, because there is no framework."""
        if not self.checkpoint_dir:
            return
        directory = Path(self.checkpoint_dir).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "model.pt"
        torch.save(cast(nn.Module, self.model).state_dict(), path)
        logger.info(f"Saved weights to {path}")

    def _live_trackers(self) -> List[Tracker]:
        """Normalize and flow the wired trackers ONCE, writing the result back to the slot.

        Written back rather than returned fresh each call: ``flow()`` on a LIVE object returns
        that same object (idempotent), but a deferred ``Fluid`` would be REBUILT on every call —
        two MLflow runs for one training run. One attribute, flowed in place, is the same idiom
        this class already uses for ``model`` / ``loss_fn`` / ``predictions_sink``.
        """
        # `flow()` resolves a marker to the live object, which the Lazy-typed slot cannot
        # express — the cast says what the call guarantees.
        self.trackers = [flow(t) for t in as_tracker_list(self.trackers) or []]
        return cast(List[Tracker], self.trackers)

    def _log_run_context(self, trackers: List[Tracker]) -> Any:
        """Provenance through marainer's plain FUNCTIONS — hookup style (1).

        There is no callback protocol to hang a ``TrackingCallback`` on, which is exactly the
        case marainer's function-level API exists for.

        ``runnable=self`` rather than a hand-written params dict: marainer derives the
        hyperparameters from the flowed object graph and dumps the resolved config, so this
        backend and the Lightning one cannot answer "what were the hyperparameters?"
        differently. The six keys this used to spell out are a subset of what it now records.
        Returns the run manifest so ``fit()`` can finish it.
        """
        if not trackers:
            return None
        from marainer.tracking.provenance import log_run_context

        return log_run_context(trackers, {"device": str(self._device)}, runnable=self)

    def _move_metrics(self) -> None:
        """Metric state lives on the same device as the logits it consumes."""
        for collection in (self.train_metrics, self.val_metrics, self.test_metrics):
            if collection is not None:
                collection.to(self._device)

    # ---- @outputs ------------------------------------------------------------ #
    @property
    @output
    def trained_model(self) -> nn.Module:
        """The live trained model (post-run ``@output``). Raises while still deferred."""
        if isinstance(self.model, Fluid):
            raise RuntimeError("trained_model is only available after run() — the model is still deferred.")
        return cast(nn.Module, self.model)

    @property
    @output
    def computed_metrics(self) -> Dict[str, float]:
        """Final scalar metrics from the last fit/evaluate."""
        return dict(self._computed_metrics)

    @property
    @output
    def class_names(self) -> List[str]:
        """Resolved class names for the run (post-run ``@output``), or ``[]``."""
        if self.label_map is not None:
            return list(self.label_map.class_names)
        return list(self._class_names or [])
