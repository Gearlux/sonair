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
    train_set: !class:recordstream.sources.HuggingFaceSource { path: kitofrank/RFUAV, split: train }
    val_set: !class:recordstream.sources.HuggingFaceSource { path: kitofrank/RFUAV, split: validation }
    batch_size: 32
    max_epochs: 20

    recordstream run train.yaml
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Union, cast

import pytorch_lightning as L
import torch
import torch.nn as nn
from confluid import Fluid, LazyClass, Mandatory, configurable, flow, get_hierarchy_from_instance, output
from loggair import get_logger
from marainer.lightning import (
    TrackingCallback,
    activate_for_lightning,
    attach_lightning_progress,
    build_metric_collection,
    coerce_metric_value,
    scalar_metrics,
)
from marainer.outputs import ClassificationOutput
from marainer.run_names import generate_run_name
from marainer.torch import apply_class_weights, ensure_record_dataset, loss_accepts_weight
from marainer.torch.outputs import classification_output
from marainer.tracking.base import Tracker
from pytorch_lightning.loggers import Logger
from recordstream import (
    LabelMap,
    Record,
    Stream,
    batch_metadata,
    batch_tensor,
    batch_values,
    collate_records,
    is_class_id,
    iter_key,
)
from recordstream.runnable import ProgressReporting, TorchRunner, entrypoint
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MultilabelAccuracy, MultilabelF1Score
from tqdm import tqdm

logger = get_logger(__name__)

#: Which entry point :meth:`LightningClassifier.run` dispatches to.
ClassifierTask = Literal["fit", "evaluate", "test", "predict"]


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


def encode_stream(source: Any, label_map: LabelMap) -> Stream:
    """Wrap ``source`` in a :class:`~recordstream.Stream` applying ``label_map``'s encode op.

    Exposes ``label_names`` on the returned Stream so :meth:`LightningClassifier._write_class_names`
    persists ``class_names.json`` (reloaded on the eval/predict path via ``LabelMap.load``).
    """
    stream = Stream(source=flow(source), ops=[label_map.encode_op()])
    stream.label_names = label_map.label_names  # type: ignore[attr-defined]
    return stream


def _batch_target(
    batch: Record,
    key: str,
    device: Any = None,
    num_classes: Optional[int] = None,
    multilabel: bool = False,
) -> torch.Tensor:
    """The CLASSIFICATION target tensor under ``key`` in a batched record.

    Single-label: ``[N]`` int64 class ids. Multi-label (``multilabel=True``): ``[N, num_classes]``
    FLOAT multi-hot — the shape the multi-label losses take (``BCEWithLogitsLoss`` /
    ``MultiLabelSoftMarginLoss`` / ``SigmoidFocalLoss`` all want float targets shaped like the
    logits, not int64 ids).

    Reading the batch is the generic part and lives in recordstream
    (:func:`~recordstream.batch_values` / :func:`~recordstream.batch_tensor`, the inverse of
    ``collate_records``); this function is only the classification-specific SHAPING on top —
    which is exactly why it stays here rather than moving down with the readers: a segmenter's
    ``[N, H, W]`` mask and a detector's variable-N boxes start from the same values and need a
    different shape, and one shared function would just be a task switch.
    """
    value = batch_values(batch, key)
    if multilabel:
        if not num_classes:
            raise ValueError("_batch_target: multilabel targets need num_classes to build the multi-hot tensor.")
        rows = value if isinstance(value, list) else [value]
        multi_hot = torch.zeros(len(rows), int(num_classes), dtype=torch.float32)
        for row, ids in enumerate(rows):
            for class_id in ids if isinstance(ids, (list, tuple, set)) else [ids]:
                index = int(class_id)
                if 0 <= index < num_classes:
                    multi_hot[row, index] = 1.0
        return multi_hot if device is None else multi_hot.to(device)

    return batch_tensor(batch, key, device)


def _per_record_predictions(preds: Any) -> List[Any]:
    """Split a model's prediction output into ONE entry per record.

    The sink contract (``write(prediction, metadata)``) is per-record, but a classification
    model emits the whole batch at once. Handled shapes, in order: a list (already per-record),
    a ``{"predictions": [...]}`` wrapper (a detector's convention), a batched mapping such as
    :class:`~marainer.outputs.ClassificationOutput` (sliced row-wise into N mappings), and
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
        max_epochs / batch_size / num_workers / logger: ergonomic flat-config knobs.
        experiment_name / run_name: tracking identity.
        input_key: Record key carrying the model input image (default ``"image"``).
        target_key: Record key carrying the classification target (default ``"class"``).
        label_map: Optional pinned name→id :class:`~recordstream.LabelMap` (fitted from the train set
            for string targets when omitted; reloaded from a checkpoint-adjacent ``class_names.json``
            on the eval path).
        balance_classes: Inject inverse-frequency class weights into a weight-capable loss (``fit``).
        metrics: Single-metric templates flowed per split (default top-1 accuracy + macro F1).

    Post-construction slots (YAML / broadcasting): ``val_set``, ``optimizer``, ``train/val/test_loader``,
    ``lightning``, ``loss_fn``, ``predictions_sink``, ``metrics_sidecar_path``, ``class_names``,
    ``trackers`` (optional marainer :class:`~marainer.tracking.base.Tracker` list), ``log_system_metrics``.
    """

    def __init__(
        self,
        model: Mandatory[nn.Module],
        train_set: Optional[Union[Dataset[Any], Fluid]] = None,
        test_set: Optional[Union[Dataset[Any], Fluid]] = None,
        num_classes: Optional[int] = None,
        task: ClassifierTask = "fit",
        max_epochs: int = 10,
        batch_size: int = 32,
        num_workers: int = 0,
        logger: Optional[Union[Logger, Iterable[Logger], bool]] = None,
        experiment_name: str = "sonair_classification",
        run_name: Optional[str] = None,
        input_key: str = "image",
        target_key: str = "class",
        label_map: Optional[LabelMap] = None,
        balance_classes: bool = True,
        metrics: Optional[dict[str, Metric | LazyClass[Metric]] | MetricCollection] = None,
    ) -> None:
        super().__init__()
        self.model: Any = model
        self.train_set: Any = train_set
        self.val_set: Any = None
        self.test_set: Any = test_set
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
        # Deferred runtime slots (flowed in the run methods) — standard Lightning defaults.
        # Sentinels: the right loss/metrics depend on whether the targets are single- or
        # multi-label, which is only known once `initialize()` peeks the dataset. `None` means
        # "not chosen" — a live default here could not be told apart from a user's explicit
        # CrossEntropyLoss, so a multi-label config would train silently wrong.
        self.loss_fn: Any = None
        self.optimizer: Any = LazyClass(torch.optim.AdamW, lr=1e-4, weight_decay=0.05)
        loader_kw = dict(
            collate_fn=collate_records,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers != 0,
        )
        self.train_loader: Any = LazyClass(DataLoader, shuffle=True, **loader_kw)
        self.val_loader: Any = LazyClass(DataLoader, shuffle=False, **loader_kw)
        self.test_loader: Any = LazyClass(DataLoader, shuffle=False, **loader_kw)
        self.lightning: Any = LazyClass(L.Trainer, max_epochs=max_epochs, logger=logger)
        self.metrics: Any = metrics
        #: True once `initialize()` sees multi-label targets. Drives the loss/metric defaults,
        #: the multi-hot collate, and whether metrics size by `num_labels` or `num_classes`.
        self.multilabel: bool = False
        self.train_metrics: Any = None
        self.val_metrics: Any = None
        self.test_metrics: Any = None
        self.predictions_sink: Any = None
        self.metrics_sidecar_path: Optional[str] = None
        self.class_names: Optional[Union[List[str], str]] = None
        self.trackers: Optional[List[Tracker]] = None
        self.log_system_metrics: bool = True
        self._label_names: Optional[List[str]] = None
        self._trackers: List[Tracker] = []

    @property
    def __torch_runner__(self) -> bool:  # type: ignore[override]  # dynamic per-task override of the marker
        """Only ``fit`` needs autograd; evaluate / test / predict are inference-only."""
        return self.task == "fit"

    # ---- run dispatch ------------------------------------------------------- #
    def run(self) -> None:
        self.initialize()
        dispatch = {"fit": self.fit, "evaluate": self.evaluate, "test": self.test, "predict": self.predict}
        entry = dispatch.get(self.task)
        if entry is None:
            raise ValueError(f"Unknown task {self.task!r}; expected one of {list(dispatch)}.")
        entry()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, cast(nn.Module, self.model)(x))

    # ---- standard Lightning step functions ---------------------------------- #
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        self._resolve_task_defaults()  # no-op once resolved; keeps a directly-called step working
        x = batch_tensor(batch, self.input_key, self.device)
        y = _batch_target(batch, self.target_key, self.device, self.num_classes, self.multilabel)
        preds = self(x)
        loss: torch.Tensor = self.loss_fn(preds, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=x.shape[0])
        if isinstance(self.train_metrics, MetricCollection):
            self.train_metrics(preds, y)
            self.log_dict(self.train_metrics, on_step=True, on_epoch=False)
        return loss

    def on_train_epoch_end(self) -> None:
        if isinstance(self.train_metrics, MetricCollection):
            self.train_metrics.reset()

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        self._resolve_task_defaults()  # no-op once resolved; keeps a directly-called step working
        x = batch_tensor(batch, self.input_key, self.device)
        y = _batch_target(batch, self.target_key, self.device, self.num_classes, self.multilabel)
        preds = self(x)
        loss: torch.Tensor = self.loss_fn(preds, y)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, batch_size=x.shape[0])
        if isinstance(self.val_metrics, MetricCollection):
            self.val_metrics.update(preds, y)
            self.log_dict(self.val_metrics, on_step=False, on_epoch=True, batch_size=x.shape[0])
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        x = batch_tensor(batch, self.input_key, self.device)
        preds = self(x)
        if isinstance(self.test_metrics, MetricCollection):
            self.test_metrics.update(
                preds, _batch_target(batch, self.target_key, self.device, self.num_classes, self.multilabel)
            )
            self.log_dict(self.test_metrics, on_step=False, on_epoch=True, batch_size=x.shape[0])
        if self.predictions_sink is not None:
            self._write_predictions(classification_output(preds), batch)
        return preds

    def predict_step(self, batch: Any, batch_idx: int) -> ClassificationOutput[torch.Tensor]:
        """Predict a batch as a :class:`~marainer.outputs.ClassificationOutput`.

        Returning the CONTRACT rather than raw logits is what makes the output consumable:
        a predictions sink reads ``probs`` / ``class_idx`` by name (it cannot softmax on the
        caller's behalf — only the model knows whether its head already did).
        """
        out = classification_output(self(batch_tensor(batch, self.input_key, self.device)))
        if self.predictions_sink is not None:
            self._write_predictions(out, batch)
        return out

    def configure_optimizers(self) -> Any:
        return flow(self.optimizer, params=self.parameters())

    # ---- the four entry points ---------------------------------------------- #
    @entrypoint("fit", role="trainer", primary=True)
    def fit(self) -> None:
        logger.info(f"sonair training: {self.experiment_name} ({self.run_name})")
        if self.train_set is None:
            raise ValueError("LightningClassifier.fit() requires a 'train_set'.")
        train_ds = ensure_record_dataset(flow(self.train_set))
        val_ds = ensure_record_dataset(flow(self.val_set)) if self.val_set is not None else None
        test_ds = ensure_record_dataset(flow(self.test_set)) if self.test_set is not None else None
        t_loader = flow(self.train_loader, dataset=train_ds)
        v_loader = flow(self.val_loader, dataset=val_ds) if val_ds is not None else None
        trainer = self._build_trainer()
        trainer.fit(model=self, train_dataloaders=t_loader, val_dataloaders=v_loader)
        logger.info("Training complete.")
        self._label_names = _resolve_label_names(train_ds)
        if self._label_names:
            self._write_class_names(trainer, self._label_names)
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
        dataset = ensure_record_dataset(flow(self.test_set))
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
        dataset = ensure_record_dataset(flow(source))
        if self.predictions_sink is not None:
            self.predictions_sink = flow(self.predictions_sink)
        loader = flow(getattr(self, loader_slot), dataset=dataset)
        trainer = self._build_trainer()
        runner = trainer.validate if stage == "validate" else trainer.test
        runner(model=self, dataloaders=loader)
        if self.predictions_sink is not None and hasattr(self.predictions_sink, "close"):
            self.predictions_sink.close()
        self._finalize_metrics(getattr(self, metrics_slot))

    def _build_trainer(self) -> L.Trainer:
        """Flow the ``pl.Trainer`` and attach the OPTIONAL marainer tracking (a few lines; removable)."""
        trainer = cast(L.Trainer, flow(self.lightning))
        if self.trackers:
            self._trackers = list(self.trackers)
            cb = TrackingCallback(self._trackers, params=self._config_params())
            trainer.callbacks.append(cb)  # type: ignore[attr-defined]
        attach_lightning_progress(trainer, self)
        if self.log_system_metrics:
            activate_for_lightning(trainer)
        return trainer

    def _config_params(self) -> Dict[str, Any]:
        """Flat ``{dotted-path: live value}`` of this runnable's configuration, for tracker params.

        Walks the FLOWED object graph (not the static signature), so it sees defaults the user
        never wrote into YAML plus post-construction keys. ``log_run_context`` keeps only the
        JSON-scalar values, so nested objects here are harmless. Best-effort by design: a
        tracking detail must never take down a training run.
        """
        try:
            return {path: value for path, (_, value, _) in get_hierarchy_from_instance(self).items()}
        except Exception:  # pragma: no cover - best effort
            return {}

    # ---- classification specifics ------------------------------------------- #
    def initialize(self) -> None:
        """Resolve the label encoding + ``num_classes``, then flow the model + metrics with it."""
        if self.task == "fit":
            self._initialize_fit()
        else:
            self._initialize_eval()

    def _initialize_fit(self) -> None:
        self.train_set = ensure_record_dataset(flow(self.train_set))
        if self.val_set is not None:
            self.val_set = ensure_record_dataset(flow(self.val_set))
        if self.test_set is not None:
            self.test_set = ensure_record_dataset(flow(self.test_set))

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
        want_weights = self.balance_classes and loss_accepts_weight(self.loss_fn)
        need_fit = targets_are_names and self.label_map is None
        raw_targets: Optional[List[Any]] = None
        if need_fit or want_weights or nc is None:
            raw_targets = self._walk_targets(self.train_set)

        if need_fit:
            assert raw_targets is not None
            self.label_map = LabelMap.fit(raw_targets)
            logger.info(f"Fitted LabelMap with {self.label_map.num_classes} classes from train targets.")
        if targets_are_names and self.label_map is not None:
            self.train_set = encode_stream(self.train_set, self.label_map)
            if self.val_set is not None:
                self.val_set = encode_stream(self.val_set, self.label_map)
            if self.test_set is not None:
                self.test_set = encode_stream(self.test_set, self.label_map)

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
            self._apply_class_weights(self.num_classes, raw_targets)

        self.model = flow(self.model, num_classes=self.num_classes)
        if self.train_metrics is None:
            base = build_metric_collection(self.metrics, **self._metric_dims())
            self.train_metrics = base.clone(prefix="train/")
            self.val_metrics = base.clone(prefix="val/")
            self.test_metrics = base.clone(prefix="test/")

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
        dataset = ensure_record_dataset(flow(source))
        first = _first_target(dataset, self.target_key)
        self.multilabel = isinstance(first, (list, tuple, set))
        self._resolve_task_defaults()
        probe = next(iter(first), None) if self.multilabel else first
        if probe is not None and not is_class_id(probe):
            label_map = self.label_map or self._load_pinned_label_map()
            if label_map is None:
                raise RuntimeError("eval dataset has string targets but no label_map and no class_names.json found.")
            self.label_map = label_map
            dataset = encode_stream(dataset, label_map)
        setattr(self, eval_slot, dataset)
        metrics_slot = "val_metrics" if self.task == "evaluate" else "test_metrics"
        prefix = "val/" if self.task == "evaluate" else "test/"
        if getattr(self, metrics_slot) is None:
            base = build_metric_collection(self.metrics, **self._metric_dims())
            setattr(self, metrics_slot, base.clone(prefix=prefix))

    def _walk_targets(self, source: Any) -> List[Any]:
        src = flow(source)
        total = len(src) if hasattr(src, "__len__") else None
        return list(tqdm(iter_key(src, self.target_key), total=total, desc="sonair: reading targets", unit="record"))

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

    def _metric_dims(self) -> dict[str, int]:
        """The sizing kwarg metric templates are flowed with.

        torchmetrics sizes multi-label metrics by ``num_labels`` and multi-class ones by
        ``num_classes`` — same number, different keyword.
        """
        assert self.num_classes is not None
        return {"num_labels": self.num_classes} if self.multilabel else {"num_classes": self.num_classes}

    def _apply_class_weights(self, num_classes: int, raw_targets: Optional[List[Any]]) -> None:
        """Encode the walked targets to class ids, then hand the policy to marainer.

        Only the ENCODING is sonair's — it depends on this trainer's
        :class:`~recordstream.LabelMap`. The capability check, the
        inverse-frequency arithmetic and the injection are engine-generic and
        live in :mod:`marainer.torch.balancing`, so a segmentation or detection
        trainer gets them without reimplementing anything.
        """
        # `raw_targets` is None when no walk was needed — which, at this call site,
        # means the loss is not weight-capable (`want_weights` gated the walk). Pass
        # the empty set through rather than asserting: `apply_class_weights` does the
        # capability check itself and emits the "loss has no 'weight' arg" warning,
        # so the diagnostic stays in one place instead of being duplicated here.
        # `to_ids` maps names through the fitted map AND passes already-encoded ids
        # through, so there is no name-vs-id branch here at all. An empty LabelMap is
        # valid for the id case — that is the contract.
        label_map = self.label_map or LabelMap()
        targets = [i for t in (raw_targets or []) if t is not None for i in label_map.to_ids(t)]
        # Rebind: a deferred `!class:` loss is FLOWED with the weights, so the
        # returned object is not the one passed in.
        self.loss_fn, self.class_weights = apply_class_weights(self.loss_fn, targets, num_classes)

    def _load_pinned_label_map(self) -> Optional[LabelMap]:
        ckpt_path = getattr(self.model, "checkpoint_path", None)
        if not ckpt_path:
            return None
        candidate = Path(str(ckpt_path)).expanduser().parent / "class_names.json"
        if not candidate.is_file():
            return None
        try:
            label_map = LabelMap.load(candidate)
            logger.info(f"Loaded pinned LabelMap ({label_map.num_classes} classes) from {candidate}.")
            return label_map
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Could not load LabelMap from {candidate}: {exc}")
            return None

    # ---- predictions + tracking-side helpers -------------------------------- #
    def _write_predictions(self, preds: Any, batch: Any) -> None:
        """Write ONE prediction per record to the wired sink, paired with that record's metadata.

        The model emits a BATCHED output (``probs`` ``[N, C]``, ``class_idx`` ``[N]``) while the
        sink contract is per-record — so a batched :class:`~marainer.outputs.ClassificationOutput`
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
            self.predictions_sink.write(item, m)

    def _write_class_names(self, trainer: L.Trainer, label_names: List[str]) -> None:
        payload = {"class_names": label_names, "num_classes": len(label_names)}
        cb = getattr(trainer, "checkpoint_callback", None)
        dirpath = getattr(cb, "dirpath", None) if cb is not None else getattr(trainer, "default_root_dir", None)
        if dirpath:
            out_path = Path(dirpath)
            out_path.mkdir(parents=True, exist_ok=True)
            fname = out_path / "class_names.json"
            fname.write_text(json.dumps(payload, indent=2, sort_keys=True))
            logger.info(f"Wrote class_names.json to {fname}")
            for tracker in self._trackers:
                tracker.log_artifact(fname)

    def _finalize_metrics(self, metrics: Any) -> None:
        if not isinstance(metrics, MetricCollection) or self.metrics_sidecar_path is None:
            return
        try:
            payload = coerce_metric_value(metrics.compute())
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Could not compute final metrics for sidecar: {exc}")
            return
        wrapped = {"experiment_name": self.experiment_name, "run_name": self.run_name, "metrics": payload}
        out_path = Path(self.metrics_sidecar_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(wrapped, indent=2, sort_keys=True))
        logger.info(f"Wrote final metrics sidecar to {out_path}")

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
    def label_names(self) -> List[str]:
        """Resolved class/label names for the run (post-run ``@output``), or ``[]``."""
        if self.label_map is not None:
            return list(self.label_map.label_names)
        return list(self._label_names or [])


def _run_prefix(task: str) -> str:
    return {"fit": "train", "predict": "predict"}.get(task, "evaluate")


def _resolve_label_names(dataset: Any) -> Optional[List[str]]:
    names = getattr(dataset, "label_names", None)
    if not names:
        return None
    try:
        return [str(n) for n in names]
    except TypeError:
        return None


__all__ = ["ClassifierTask", "LightningClassifier", "encode_stream"]
