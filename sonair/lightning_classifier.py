"""LightningClassifier: a standard PyTorch-Lightning image classifier (train + eval in one class).

A self-contained ``pytorch_lightning.LightningModule`` — its own ``training_step`` /
``validation_step`` / ``test_step`` / ``predict_step`` / ``configure_optimizers``, and a
``run()`` that builds the loaders + ``pl.Trainer`` and calls ``fit`` / ``test`` / ``predict``
per the ``task`` knob. It reads like a textbook Lightning example; marainer is used ONLY for
optional experiment tracking (git / config / params / system metrics / log file) and a couple
of small utilities (``ensure_record_dataset``, ``build_metric_collection``). Remove marainer and
this class still trains — you lose the rich tracking, not the training.

Classification specifics it owns: deriving ``num_classes`` (from a concrete model head, an
explicit arg, a ``LabelMap``, or a dataset walk), fitting/pinning a ``LabelMap`` for string
targets, optional inverse-frequency class balancing, and building the per-split metric collections.

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

import inspect
from typing import Any, Dict, Iterable, List, Literal, Optional, Union, cast

import numpy as np
import pytorch_lightning as L
import torch
import torch.nn as nn
from confluid import Fluid, Lazy, LazyClass, Mandatory, configurable, flow, output
from confluid.registry import resolve_class
from loggair import get_logger
from marainer.artifacts import load_class_names, save_class_names, write_metrics_sidecar
from marainer.lightning import (
    TrackerLogger,
    TrackingCallback,
    activate_for_lightning,
    attach_lightning_progress,
    build_metric_collection,
    checkpoint_dir,
    scalar_metrics,
)
from marainer.run_names import generate_run_name
from marainer.tracking.base import Tracker, as_tracker_list
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
from tqdm import tqdm

from sonair.metric_slots import merge_metric_specs

logger = get_logger(__name__)

#: Which entry point :meth:`LightningClassifier.run` dispatches to.
ClassifierTask = Literal["fit", "evaluate", "test", "predict"]

#: What a metrics slot admits: a ``{name: template}`` mapping, a ready-made collection, or
#: ``None`` for "not chosen". Named because FOUR slots share it (the shared set + one per split)
#: and `to_pydantic` reads the annotation for navigaitor's form-spec / StreamStudio widgets.
MetricsSpec = Optional[Union[Dict[str, Union[Metric, LazyClass[Metric]]], MetricCollection]]


def _first_target(source: Any, target_key: str) -> Any:
    """The first non-``None`` target value in ``source`` (``None`` when there are none).

    One peek is enough to know whether the targets need encoding — the ANSWER comes from
    :func:`recordstream.is_class_id`, the single rule for "is this a class id or a class name?",
    rather than a type check re-derived here.
    """
    for target in iter_key(flow(source), target_key):
        if target is not None:
            return target
    return None


def _per_record_predictions(preds: Any) -> List[Any]:
    """Split a model's prediction output into ONE entry per record.

    The sink contract (``write(prediction, metadata)``) is per-record, but a classification
    model emits the whole batch at once. Handled shapes, in order: a list (already per-record),
    a ``{"predictions": [...]}`` wrapper (a detector's convention), a batched mapping such as
    :class:`~recordstream.ClassificationOutput` (sliced row-wise into N mappings), and
    anything else (treated as ONE prediction).
    """
    if isinstance(preds, list):
        return preds
    if isinstance(preds, dict):
        if "predictions" in preds:
            return list(preds["predictions"])
        lengths = {len(v) for v in preds.values() if hasattr(v, "__len__")}
        if len(lengths) == 1:
            n = lengths.pop()
            return [{k: v[i] for k, v in preds.items()} for i in range(n)]
    return [preds]


def _run_prefix(task: str) -> str:
    return {"fit": "train", "predict": "predict"}.get(task, "evaluate")


@configurable(task="classification", role="trainer", framework="torch")
class LightningClassifier(TorchRunner, ProgressReporting, L.LightningModule):
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
        trackers: marainer trackers — one or a list. They ARE this run's metric sinks:
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
            (``marainer[plot]``) a cost of every eval epoch, where the table costs nothing.

    Post-construction slots (YAML / broadcasting): ``val_set``, ``optimizer``, ``train/val/test_loader``,
    ``lightning``, ``loss_fn``, ``predictions_sink``, ``metrics_sidecar_path``,
    ``log_system_metrics``.
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
    ) -> None:
        super().__init__()
        self.model: Lazy[nn.Module] = model
        self.train_set: Optional[Lazy[RecordSource]] = train_set
        self.val_set: Optional[Lazy[RecordSource]] = None
        self.test_set: Optional[Lazy[RecordSource]] = test_set
        self.task: ClassifierTask = task
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.experiment_name = experiment_name
        self.run_name = run_name if run_name is not None else generate_run_name(prefix=_run_prefix(task))
        self.input_key = str(input_key)
        self.target_key = str(target_key)
        self.label_map = label_map
        self.balance_classes = balance_classes
        self.class_weights: Optional[torch.Tensor] = None
        self.max_epochs = max_epochs
        # Deferred runtime slots — `Lazy[T]` DECLARES that the run methods flow this later,
        # usually with a kwarg that does not exist yet (`params=self.parameters()`, `dataset=`).
        # What that buys differs by slot kind, and the difference is worth knowing:
        #   * CONSTRUCTOR params (`model`/`train_set`/`test_set`) — the marker is machine-read:
        #     `confluid.lazy_param_names` reports them and liquifai's auto deep-flow SKIPS them
        #     instead of eagerly building them.
        #   * BODY slots (below) — `lazy_param_names` scans `__init__` PARAMS only, so the
        #     marker is inert here; what actually keeps them deferred is holding a `LazyClass`
        #     value, which every walker recognises. The annotation still earns its place: it
        #     gives `to_pydantic` a real type (navigaitor form-spec / StreamStudio widgets) where
        #     `Any` — what this class used to say — typed nothing at all.
        # Sentinels: the right loss/metrics depend on whether the targets are single- or
        # multi-label, which is only known once `initialize()` peeks the dataset. `None` means
        # "not chosen" — a live default here could not be told apart from a user's explicit
        # CrossEntropyLoss, so a multi-label config would train silently wrong.
        # `Lazy[...]`: a `!class:` loss stays deferred until `apply_class_weights` flows it with
        # `weight=` (a Fluid cannot be mutated in place), so this slot IS one this class flows.
        self.loss_fn: Optional[Lazy[nn.Module]] = None
        self.optimizer: Lazy[Optimizer] = LazyClass(torch.optim.AdamW, lr=1e-4, weight_decay=0.05)
        loader_kw = dict(
            collate_fn=collate_records,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers != 0,
        )
        self.train_loader: Lazy[DataLoader[Any]] = LazyClass(DataLoader, shuffle=True, **loader_kw)
        self.val_loader: Lazy[DataLoader[Any]] = LazyClass(DataLoader, shuffle=False, **loader_kw)
        self.test_loader: Lazy[DataLoader[Any]] = LazyClass(DataLoader, shuffle=False, **loader_kw)
        # No `logger=` here: the trackers ARE the metric sinks, adapted to Lightning's logger
        # interface in `_build_trainer` (a Tracker may still be a deferred Fluid at this point,
        # so the adapter is built at run time, not now).
        self.lightning: Lazy[L.Trainer] = LazyClass(L.Trainer, max_epochs=max_epochs)
        self.metrics: Any = metrics
        #: True once `initialize()` sees multi-label targets. Drives the loss/metric defaults,
        #: the multi-hot collate, and whether metrics size by `num_labels` or `num_classes`.
        self.multilabel: bool = False
        # Per-split slots, each dual-purpose the way `model` is: they hold that split's metrics
        # SPEC until `initialize()` builds it, and the live prefixed `MetricCollection` after.
        # `_metrics_built` is what tells the two states apart — `is None` cannot, once a slot
        # may legitimately arrive carrying a spec.
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
        self.log_system_metrics: bool = True
        self._class_names: Optional[List[str]] = None

    @property
    def __needs_autograd__(self) -> bool:  # type: ignore[override]  # dynamic per-task override of the marker
        """Only ``fit`` needs autograd; evaluate / test / predict are inference-only."""
        return self.task == "fit"

    # ---- run dispatch ------------------------------------------------------- #
    def run(self) -> None:
        self.initialize()
        # The `@entrypoint` markers below ARE the dispatch table — a hand-written
        # {task: method} dict restates them and drifts (recordstream `docs/runnable.md`).
        run_entrypoint(self, self.task)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, cast(nn.Module, self.model)(x))

    # ---- standard Lightning step functions ---------------------------------- #
    def _targets(self, batch: Any) -> torch.Tensor:
        """The target tensor this task's loss expects — the ONLY task-shaped choice left.

        Both branches are recordstream calls; what is classification-specific is WHICH one and
        the dtype named for it. Multi-label wants an ``[N, C]`` float multi-hot (what
        ``BCEWithLogitsLoss`` and friends consume), single-label an ``[N]`` int64 class-id
        vector — ``int64`` explicitly, because a dataset yielding int32 label tensors is legal
        and ``CrossEntropyLoss`` rejects it with *"expected target dtype to be Long or Byte, but got Int"*.
        """
        if self.multilabel:
            if not self.num_classes:
                raise RuntimeError("multi-label targets need num_classes — call initialize() first.")
            return torch.as_tensor(multi_hot(batch, self.target_key, self.num_classes)).to(self.device)
        return batch_tensor(batch, self.target_key, self.device, dtype=torch.int64)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        self._resolve_task_defaults()  # no-op once resolved; keeps a directly-called step working
        x = batch_tensor(batch, self.input_key, self.device)
        y = self._targets(batch)
        preds = self(x)
        loss: torch.Tensor = cast(nn.Module, self.loss_fn)(preds, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=x.shape[0])
        if isinstance(self.train_metrics, MetricCollection):
            # The per-BATCH values, SCALARS only — `log_dict` cannot carry the rest (see
            # `_report_epoch_metrics`), and a per-step confusion matrix says nothing anyway.
            self.log_dict(scalar_metrics(self.train_metrics(preds, y)), on_step=True, on_epoch=False)
        return loss

    def on_train_epoch_end(self) -> None:
        if isinstance(self.train_metrics, MetricCollection):
            self.train_metrics.reset()

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        self._resolve_task_defaults()  # no-op once resolved; keeps a directly-called step working
        x = batch_tensor(batch, self.input_key, self.device)
        y = self._targets(batch)
        preds = self(x)
        loss: torch.Tensor = cast(nn.Module, self.loss_fn)(preds, y)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, batch_size=x.shape[0])
        if isinstance(self.val_metrics, MetricCollection):
            self.val_metrics.update(preds, y)
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        x = batch_tensor(batch, self.input_key, self.device)
        preds = self(x)
        if isinstance(self.test_metrics, MetricCollection):
            self.test_metrics.update(preds, self._targets(batch))
        if self.predictions_sink is not None:
            self._write_predictions(classification_output(preds), batch)
        return preds

    def predict_step(self, batch: Any, batch_idx: int) -> ClassificationOutput[torch.Tensor]:
        """Predict a batch as a :class:`~recordstream.ClassificationOutput`.

        Returning the CONTRACT rather than raw logits is what makes the output consumable:
        a predictions sink reads ``probs`` / ``class_idx`` by name (it cannot softmax on the
        caller's behalf — only the model knows whether its head already did).
        """
        out = classification_output(self(batch_tensor(batch, self.input_key, self.device)))
        if self.predictions_sink is not None:
            self._write_predictions(out, batch)
        return out

    # ---- epoch-end metric reporting ----------------------------------------- #
    # An eval split's metrics are computed HERE rather than by handing the whole
    # `MetricCollection` to `log_dict`, and the reason is that `log_dict` carries SCALARS
    # only: Lightning reduces every logged value through `convert_tensors_to_scalars`, which
    # raises *"does not contain a single element"* on an N×N confusion matrix — so a config
    # wiring one used to kill the validation loop on the first epoch (measured). Which
    # metrics are scalar cannot be decided when they are BUILT — it is a property of the
    # COMPUTED value, and a fresh `MulticlassAUROC.compute()` raises where a fresh
    # `MulticlassConfusionMatrix.compute()` returns a matrix, so no probe tells them apart
    # up front. Computing once at epoch end does: the scalars go to `log_dict` exactly as
    # before, the matrices are rendered.
    #
    # Reset happens at epoch START, not end, so the LAST epoch's state survives the run —
    # `_write_metrics_sidecar` and the `computed_metrics` @output both `compute()` after
    # `trainer.validate()` / `.test()` has returned.
    def on_validation_epoch_start(self) -> None:
        if isinstance(self.val_metrics, MetricCollection):
            self.val_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        self._report_epoch_metrics(self.val_metrics)

    def on_test_epoch_start(self) -> None:
        if isinstance(self.test_metrics, MetricCollection):
            self.test_metrics.reset()

    def on_test_epoch_end(self) -> None:
        self._report_epoch_metrics(self.test_metrics)

    def _report_epoch_metrics(self, collection: Any) -> None:
        """Compute one split's metrics: scalars to ``log_dict``, confusion matrices reported.

        Which computed values ARE confusion matrices is recordstream's rule
        (:func:`~recordstream.ops.image.confusion_matrices_payload` — square 2-D, by SHAPE not
        by metric name), and the renderers are marainer's; this method only decides WHEN and
        WHERE — the log always, plus every wired tracker's text channel, which is what puts the
        matrix in MLflow / TensorBoard / Aim / ClearML beside the scalars, plus its figure
        channel when :attr:`plot_metrics` asks for one.
        """
        if not isinstance(collection, MetricCollection):
            return
        values = collection.compute()
        self.log_dict(scalar_metrics(values))
        if self._trainer is not None and self.trainer.sanity_checking:
            return  # two batches through an untrained model: a matrix nobody wants to read
        step = self.current_epoch if self._trainer is not None else None
        trackers = self._live_trackers()
        for payload in confusion_matrices_payload(values, self.class_names or None):
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

    def configure_optimizers(self) -> Any:
        return flow(self.optimizer, params=self.parameters())

    # ---- the four entry points ---------------------------------------------- #
    @entrypoint("fit", role="trainer", primary=True)
    def fit(self) -> None:
        logger.info(f"sonair training: {self.experiment_name} ({self.run_name})")
        if self.train_set is None:
            raise ValueError("LightningClassifier.fit() requires a 'train_set'.")
        train_ds = ensure_record_dataset(self.train_set)
        val_ds = ensure_record_dataset(self.val_set) if self.val_set is not None else None
        test_ds = ensure_record_dataset(self.test_set) if self.test_set is not None else None
        t_loader = flow(self.train_loader, dataset=train_ds)
        v_loader = flow(self.val_loader, dataset=val_ds) if val_ds is not None else None
        trainer = self._build_trainer()
        trainer.fit(model=self, train_dataloaders=t_loader, val_dataloaders=v_loader)
        logger.info("Training complete.")
        self._class_names = class_names(train_ds, val_ds, test_ds)
        save_class_names(self._class_names, checkpoint_dir(trainer), self._live_trackers())

        if test_ds is not None:
            trainer.test(model=self, dataloaders=flow(self.test_loader, dataset=test_ds))

    @entrypoint("evaluate", role="evaluator")
    def evaluate(self) -> None:
        self._run_eval("validate", "val_set", "val_loader", "val_metrics")

    @entrypoint("test", role="evaluator", primary=True)
    def test(self) -> None:
        self._run_eval("test", "test_set", "test_loader", "test_metrics")

    @entrypoint("predict", role="predictor", primary=True)
    def predict(self) -> None:
        logger.info(f"sonair predicting: {self.experiment_name} ({self.run_name})")
        if self.test_set is None:
            raise ValueError("LightningClassifier.predict() requires a 'test_set'.")
        dataset = ensure_record_dataset(self.test_set)
        if self.predictions_sink is not None:
            self.predictions_sink = flow(self.predictions_sink)
        loader = flow(self.test_loader, dataset=dataset)
        trainer = self._build_trainer()
        trainer.predict(model=self, dataloaders=loader)
        if self.predictions_sink is not None and hasattr(self.predictions_sink, "close"):
            self.predictions_sink.close()

    def _run_eval(self, stage: str, dataset_slot: str, loader_slot: str, metrics_slot: str) -> None:
        logger.info(f"sonair evaluating ({stage}): {self.experiment_name} ({self.run_name})")
        source = getattr(self, dataset_slot)
        if source is None:
            raise ValueError(f"LightningClassifier.{stage}() requires a '{dataset_slot}'.")
        dataset = ensure_record_dataset(source)
        if self.predictions_sink is not None:
            self.predictions_sink = flow(self.predictions_sink)
        loader = flow(getattr(self, loader_slot), dataset=dataset)
        trainer = self._build_trainer()
        runner = trainer.validate if stage == "validate" else trainer.test
        runner(model=self, dataloaders=loader)
        if self.predictions_sink is not None and hasattr(self.predictions_sink, "close"):
            self.predictions_sink.close()
        self._write_metrics_sidecar(getattr(self, metrics_slot))

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

    def _build_trainer(self) -> L.Trainer:
        """Flow the ``pl.Trainer`` and attach the OPTIONAL marainer tracking (a few lines; removable).

        The trackers ARE the Trainer's loggers: each is flowed (it may still be a deferred
        ``Fluid``) and adapted through ``TrackerLogger``, so Lightning's per-step metrics land
        in the same backends as the provenance and the manifest. There is no separate
        ``logger:`` slot — one wiring, one place a run's output goes.
        """
        trackers = self._live_trackers()
        adapters = [TrackerLogger(t) for t in trackers]
        # A runtime kwarg wins over the stored one, so an explicitly wired `lightning:` block
        # keeps every knob it set and only gains the loggers.
        trainer = cast(L.Trainer, flow(self.lightning, logger=adapters or False))
        if trackers:
            # `runnable=self` rather than a params dict: marainer derives the hyperparameters
            # from the flowed object graph AND dumps the resolved config, so this backend and
            # the torch one cannot answer "what were the hyperparameters?" differently.
            cb = TrackingCallback(trackers, runnable=self)
            trainer.callbacks.append(cb)  # type: ignore[attr-defined]
        attach_lightning_progress(trainer, self)
        if self.log_system_metrics:
            activate_for_lightning(trainer)
        return trainer

    # ---- classification specifics ------------------------------------------- #
    def initialize(self) -> None:
        """Resolve the label encoding + ``num_classes``, then flow the model + metrics with it."""
        if self.task == "fit":
            self._initialize_fit()
        else:
            self._initialize_eval()

    def _initialize_fit(self) -> None:
        self.train_set = ensure_record_dataset(self.train_set)
        if self.val_set is not None:
            self.val_set = ensure_record_dataset(self.val_set)
        if self.test_set is not None:
            self.test_set = ensure_record_dataset(self.test_set)

        # `iter_key` unwraps a MultiLabel to its list of values, so a sequence here IS a
        # multi-label target — the item type, not a guess about what a list might mean.
        first = _first_target(self.train_set, self.target_key)
        self.multilabel = isinstance(first, (list, tuple, set))

        # Names need a LabelMap; ids do not. `is_class_id` is recordstream's rule (bool excluded,
        # so a boolean flag wired to the target key cannot silently become class 1). For a
        # multi-label target the question is asked of its FIRST member.
        probe = next(iter(first), None) if self.multilabel else first
        targets_are_names = probe is not None and not is_class_id(probe)
        # Resolve the sentinels HERE: the capability probe below reads `loss_fn`, and an
        # unresolved `None` would report "cannot be balanced" and skip the target walk.
        self._resolve_task_defaults()
        nc = self._num_classes_without_data()
        want_weights = self.balance_classes and self.accepts_class_weights()
        need_fit = targets_are_names and self.label_map is None
        raw_targets: Optional[List[Any]] = None
        if need_fit or want_weights or nc is None:
            raw_targets = self._walk_targets(self.train_set)

        if need_fit:
            assert raw_targets is not None
            self.label_map = LabelMap.fit(raw_targets)
            logger.info(f"Fitted LabelMap with {self.label_map.num_classes} classes from train targets.")
        if targets_are_names and self.label_map is not None:
            self.train_set = self.label_map.encode(self.train_set)
            if self.val_set is not None:
                self.val_set = self.label_map.encode(self.val_set)
            if self.test_set is not None:
                self.test_set = self.label_map.encode(self.test_set)

        if nc is None:
            if self.label_map is not None:
                nc = self.label_map.num_classes
            else:
                assert raw_targets is not None
                # An empty map still maps ids through — that is the `to_ids` contract.
                ids = [i for t in raw_targets for i in LabelMap().to_ids(t)]
                nc = max(ids) + 1 if ids else None
        if not nc:
            raise RuntimeError(
                "could not determine num_classes — set it on the trainer/model or wire a labelled dataset."
            )
        self.num_classes = int(nc)

        if self.balance_classes and self.class_weights is None:
            self._balance_loss(self.num_classes, raw_targets)

        self.model = flow(self.model, num_classes=self.num_classes)
        if not self._metrics_built:
            self.train_metrics = self._build_split_metrics(self.train_metrics, "train/")
            self.val_metrics = self._build_split_metrics(self.val_metrics, "val/")
            self.test_metrics = self._build_split_metrics(self.test_metrics, "test/")
            self._metrics_built = True

    def _initialize_eval(self) -> None:
        # The trained model's head is authoritative for eval — flow it first, read num_classes off it.
        self.model = flow(self.model)
        nc = getattr(self.model, "num_classes", None)
        if not nc:
            raise RuntimeError("could not determine num_classes from the model (it must expose `num_classes`).")
        self.num_classes = int(nc)
        eval_slot = "val_set" if self.task == "evaluate" else "test_set"
        source = getattr(self, eval_slot)
        if source is None:
            return
        dataset = ensure_record_dataset(source)
        first = _first_target(dataset, self.target_key)
        self.multilabel = isinstance(first, (list, tuple, set))
        self._resolve_task_defaults()
        probe = next(iter(first), None) if self.multilabel else first
        if probe is not None and not is_class_id(probe):
            label_map = self.label_map or load_class_names(getattr(self.model, "checkpoint_path", None))
            if label_map is None:
                raise RuntimeError("eval dataset has string targets but no label_map and no class_names.json found.")
            self.label_map = label_map
            dataset = label_map.encode(dataset)
        setattr(self, eval_slot, dataset)
        metrics_slot = "val_metrics" if self.task == "evaluate" else "test_metrics"
        prefix = "val/" if self.task == "evaluate" else "test/"
        if not self._metrics_built:
            setattr(self, metrics_slot, self._build_split_metrics(getattr(self, metrics_slot), prefix))
            self._metrics_built = True

    def _walk_targets(self, source: Any) -> List[Any]:
        """Materialize the target column once, with a progress bar.

        Everything here except the bar is recordstream's: ``iter_key`` projects, so a
        projection-aware source never builds the values this does not ask for, and it
        materializes a deferred source itself. What remains is the PROGRESS — reading 60k
        labels is a visible pause a user needs feedback on — which is a presentation choice
        a data engine should not make (recordstream has no tqdm dependency, and its own
        progress surface is the ``ProgressReporting`` callback, not a bar).
        """
        total = len(source) if hasattr(source, "__len__") else None
        return list(tqdm(iter_key(source, self.target_key), total=total, desc="sonair: reading targets", unit="record"))

    def _num_classes_without_data(self) -> Optional[int]:
        if not isinstance(self.model, Fluid):
            model_nc = getattr(self.model, "num_classes", None)
            if model_nc:
                if self.num_classes is not None and int(self.num_classes) != int(model_nc):
                    logger.warning(
                        f"Trainer num_classes={self.num_classes} != model num_classes={model_nc}; using model's."
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

    def _resolve_task_defaults(self) -> None:
        """Fill the ``loss_fn`` / ``metrics`` sentinels once the target shape is known.

        Both defaults depend on :attr:`multilabel`, which `initialize()` derives from the data —
        so they cannot be constructor defaults. A value the user DID wire is left untouched
        (that is what the ``None`` sentinel buys: "not chosen" is distinguishable from "chosen
        to be the single-label default").
        """
        if self.loss_fn is None:
            # Multi-label needs per-class independent sigmoids; single-label a softmax over classes.
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

        The combination rule is :func:`~sonair.metric_slots.merge_metric_specs`, shared with
        every other backend so a config moving between them cannot change shape; what is torch's
        alone is the rest of this line — sizing the templates and prefixing the collection.
        """
        return build_metric_collection(merge_metric_specs(self.metrics, spec), **self._metric_dims()).clone(
            prefix=prefix
        )

    def _metric_dims(self) -> dict[str, int]:
        """The sizing kwarg metric templates are flowed with.

        torchmetrics sizes multi-label metrics by ``num_labels`` and multi-class ones by
        ``num_classes`` — same number, different keyword.
        """
        assert self.num_classes is not None
        return {"num_labels": self.num_classes} if self.multilabel else {"num_classes": self.num_classes}

    # ---- class balancing: the STATISTIC is recordstream's, the INJECTION is this backend's -- #
    def accepts_class_weights(self) -> bool:
        """Whether this backend's loss can consume per-class weights — **framework hook**.

        Answered for torch: ``torch.nn`` losses take a ``weight`` TENSOR in the constructor, so
        the question is whether the loss's signature declares one. Works on a live loss OR a
        deferred :class:`~confluid.fluid.Fluid`, which matters: :meth:`_initialize_fit` calls
        this BEFORE the loss is built to decide whether walking the whole target stream for
        weights is worth it at all.

        Another backend overrides this together with :meth:`apply_class_weights` — Keras has no
        ``weight=`` on its losses (it takes ``class_weight`` on ``fit()``), so its answer is a
        different question entirely. Returning ``False`` is always safe: the run trains
        unweighted and skips the walk.
        """
        loss_cls = resolve_class(self.loss_fn.target) if isinstance(self.loss_fn, Fluid) else type(self.loss_fn)
        if loss_cls is None:
            return False
        try:
            return "weight" in inspect.signature(loss_cls).parameters
        except (ValueError, TypeError):  # pragma: no cover - C-extension ctor without a signature
            return False

    def apply_class_weights(self, weights: "np.ndarray") -> None:
        """Inject per-class ``weights`` into the loss — **framework hook**.

        Answered for torch: the weights become a ``Tensor`` on the loss's ``weight`` argument.
        A deferred loss is FLOWED with it (a ``Fluid`` cannot be mutated in place, so the slot
        is rebound); a live loss gets the attribute set. ``self.class_weights`` keeps the vector
        for tracking.

        Another backend overrides this with its own convention — a Keras classifier would store
        ``{class_id: weight}`` for ``model.fit(class_weight=…)`` and never touch the loss. The
        WEIGHTS themselves are framework-neutral: they come from
        :func:`recordstream.inverse_frequency_weights`, a statistic over the labels, so only
        this last step differs per backend.
        """
        tensor = torch.as_tensor(weights)
        if isinstance(self.loss_fn, Fluid):
            # A Fluid cannot be mutated in place — it is flowed with `weight=` and the slot rebound.
            self.loss_fn = flow(self.loss_fn, weight=tensor)
        else:
            cast(nn.Module, self.loss_fn).weight = tensor
        self.class_weights = tensor
        logger.info(f"Balanced loss: injected inverse-frequency class weights for {len(tensor)} classes.")

    def _balance_loss(self, num_classes: int, raw_targets: Optional[List[Any]]) -> None:
        """Derive the class weights from the walked targets and hand them to the backend hook.

        `raw_targets` is None when no walk was needed — at this call site that means the loss is
        not weight-capable (`want_weights` gated the walk), so there is nothing to balance.
        Everything about the derivation is recordstream's: ``inverse_frequency_weights`` flattens
        each target through the :class:`~recordstream.LabelMap` (names encode, ids pass through,
        a multi-label target counts for every class it names) and returns the vector.
        """
        if not raw_targets:
            return
        weights = inverse_frequency_weights(raw_targets, num_classes, self.label_map)
        if weights is None:
            logger.warning("Class balancing requested but no class ids were observed; training unweighted.")
            return
        self.apply_class_weights(weights)

    # ---- predictions + tracking-side helpers -------------------------------- #
    def _write_predictions(self, preds: Any, batch: Any) -> None:
        """Write ONE prediction per record to the wired sink, paired with that record's metadata.

        The model emits a BATCHED output (``probs`` ``[N, C]``, ``class_idx`` ``[N]``) while the
        sink contract is per-record — so a batched :class:`~recordstream.ClassificationOutput`
        is sliced row-wise here. Handing the batch over whole was a real bug: the sink read
        ``probs[0]`` as if it were one record's class distribution and silently dropped the
        rest of the batch.
        """
        per_record_meta = (
            batch_metadata(batch, exclude=(self.input_key, self.target_key)) if isinstance(batch, dict) else None
        )
        items = _per_record_predictions(preds)
        for i, item in enumerate(items):
            m = per_record_meta[i] if per_record_meta and i < len(per_record_meta) else {}
            cast(PredictionsSink, self.predictions_sink).write(item, m)

    def _write_metrics_sidecar(self, metrics: Any) -> None:
        """Hand the run's final scalars to marainer, which owns where run artifacts land."""
        if not isinstance(metrics, MetricCollection):
            return
        try:
            payload = scalar_metrics(metrics.compute())
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Could not compute final metrics for the sidecar: {exc}")
            return
        write_metrics_sidecar(self.metrics_sidecar_path, payload, self.experiment_name, self.run_name)

    # ---- @outputs ----------------------------------------------------------- #
    @property
    @output
    def trained_model(self) -> nn.Module:
        """The trained/live classification model (post-run ``@output``). Raises while still deferred."""
        if isinstance(self.model, Fluid):
            raise RuntimeError("trained_model is only available after run(): the model is still a deferred Fluid.")
        return cast(nn.Module, self.model)

    @property
    @output
    def computed_metrics(self) -> Dict[str, float]:
        """Final SCALAR metrics (post-run ``@output``). Raises before ``run()``."""
        if self.task == "fit":
            try:
                return scalar_metrics(dict(self.trainer.callback_metrics))
            except RuntimeError as exc:
                raise RuntimeError("computed_metrics is only available after run().") from exc
        metrics = self.val_metrics if self.task == "evaluate" else self.test_metrics
        if not isinstance(metrics, MetricCollection):
            raise RuntimeError("computed_metrics is only available after run().")
        return scalar_metrics(metrics.compute())

    @property
    @output
    def class_names(self) -> List[str]:
        """Resolved class/label names for the run (post-run ``@output``), or ``[]``."""
        if self.label_map is not None:
            return list(self.label_map.class_names)
        return list(self._class_names or [])


__all__ = ["ClassifierTask", "LightningClassifier"]
