"""LightningClassifier: a standard PyTorch-Lightning image classifier (train + eval in one class).

A self-contained ``pytorch_lightning.LightningModule`` — its own ``training_step`` /
``validation_step`` / ``test_step`` / ``predict_step`` / ``configure_optimizers``, and a
``run()`` that builds the loaders + ``pl.Trainer`` and calls ``fit`` / ``test`` / ``predict``
per the ``task`` knob. It reads like a textbook Lightning example; marainer is used ONLY for
optional experiment tracking (git / config / params / system metrics / log file) and a couple
of small utilities (``ensure_sample_dataset``, ``build_metric_collection``). Remove marainer and
this class still trains — you lose the rich tracking, not the training.

Classification specifics it owns: deriving ``num_classes`` (from a concrete model head, an
explicit arg, a ``LabelMap``, or a dataset walk), fitting/pinning a ``LabelMap`` for string
targets, optional inverse-frequency class balancing, and building the per-split metric collections.

Minimal flat YAML (top-level keys wired by Confluid broadcasting)::

    runnable: !class:sonair.LightningClassifier
      task: fit
      run_name: efficientvit_b0
    model: !lazy:sonair.TimmModel { model_name: efficientvit_b0, pretrained: true }
    train_set: !class:sampleflux.sources.HuggingFaceSource { path: kitofrank/RFUAV, split: train }
    val_set: !class:sampleflux.sources.HuggingFaceSource { path: kitofrank/RFUAV, split: validation }
    batch_size: 32
    max_epochs: 20

    sampleflux run train.yaml
"""

import inspect
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Union, cast

import pytorch_lightning as L
import torch
import torch.nn as nn
from confluid import Fluid, LazyClass, Mandatory, configurable, flow, output
from confluid.registry import resolve_class
from loggair import get_logger
from marainer.lightning import (
    TrackingCallback,
    activate_for_lightning,
    attach_lightning_progress,
    build_metric_collection,
    coerce_metric_value,
    scalar_metrics,
)
from marainer.run_names import generate_run_name
from marainer.torch import ensure_sample_dataset
from marainer.tracking.base import Tracker
from pytorch_lightning.loggers import Logger
from sampleflux import Flux, LabelMap, iter_targets
from sampleflux.runnable import ProgressReporting, TorchRunner, entrypoint
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from tqdm import tqdm

from sonair.classification import classification_collate_fn

logger = get_logger(__name__)

#: Which entry point :meth:`LightningClassifier.run` dispatches to.
ClassifierTask = Literal["fit", "evaluate", "test", "predict"]

#: How a sample's ``target`` is shaped — drives the encode / num_classes path.
_TargetKind = Literal["int", "str", "multilabel", "empty"]


def _loss_accepts_weight(loss: Any) -> bool:
    """True when ``loss``'s CONSTRUCTOR declares a ``weight`` parameter (balance-capability check)."""
    loss_cls = resolve_class(loss.target) if isinstance(loss, Fluid) else type(loss)
    if loss_cls is None:
        return False
    try:
        return "weight" in inspect.signature(loss_cls).parameters
    except (ValueError, TypeError):  # pragma: no cover - C-extension ctor without a signature
        return False


def _target_to_int(value: Any) -> int:
    """Coerce a single scalar class id into a Python ``int`` (rejects bool)."""
    if isinstance(value, bool):
        raise TypeError(f"target {value!r} is a bool, not a class id")
    return int(value)


def detect_target_kind(source: Any) -> _TargetKind:
    """Classify a dataset's target by peeking the first non-``None`` target (cheap projection)."""
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
    """Wrap ``source`` in a :class:`~sampleflux.Flux` applying ``label_map``'s encode op.

    Exposes ``label_names`` on the returned Flux so :meth:`LightningClassifier._write_class_names`
    persists ``class_names.json`` (reloaded on the eval/predict path via ``LabelMap.load``).
    """
    flux = Flux(source=flow(source), ops=[label_map.encode_op()])
    flux.label_names = label_map.label_names  # type: ignore[attr-defined]
    return flux


@configurable(task="classification", role="trainer")
class LightningClassifier(TorchRunner, ProgressReporting, L.LightningModule):
    """A standard Lightning image classifier — train, evaluate, test, or predict via ``task``.

    Args:
        model: Classification model emitting ``[N, num_classes]`` logits (wire ``!lazy:`` so
            :meth:`initialize` flows it with the resolved num_classes at run time).
        train_set: Training dataset / Flux (for ``task="fit"``).
        test_set: Held-out eval dataset / Flux (for ``task="test"`` / ``"predict"``; also tested
            after ``fit`` when set).
        num_classes: Output class count; omit to resolve from the model or the dataset.
        task: Entry point :meth:`run` dispatches to — ``"fit"`` / ``"evaluate"`` / ``"test"`` / ``"predict"``.
        max_epochs / batch_size / num_workers / logger: ergonomic flat-config knobs.
        experiment_name / run_name: tracking identity.
        label_map: Optional pinned name→id :class:`~sampleflux.LabelMap` (fitted from the train set
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
        label_map: Optional[LabelMap] = None,
        balance_classes: bool = True,
        metrics: dict[str, Metric | LazyClass[Metric]] | MetricCollection = {
            "accuracy_top1": LazyClass[Metric](MulticlassAccuracy, top_k=1),
            "f1_macro": LazyClass[Metric](MulticlassF1Score, average="macro"),
        },
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
        self.label_map = label_map
        self.balance_classes = balance_classes
        self.class_weights: Optional[torch.Tensor] = None
        self.max_epochs = max_epochs
        # Deferred runtime slots (flowed in the run methods) — standard Lightning defaults.
        self.loss_fn: nn.Module = nn.CrossEntropyLoss()
        self.optimizer: Any = LazyClass(torch.optim.AdamW, lr=1e-4, weight_decay=0.05)
        loader_kw = dict(
            collate_fn=classification_collate_fn,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers != 0,
        )
        self.train_loader: Any = LazyClass(DataLoader, shuffle=True, **loader_kw)
        self.val_loader: Any = LazyClass(DataLoader, shuffle=False, **loader_kw)
        self.test_loader: Any = LazyClass(DataLoader, shuffle=False, **loader_kw)
        self.lightning: Any = LazyClass(L.Trainer, max_epochs=max_epochs, logger=logger)
        self.metrics = metrics
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
        preds = self(batch.input)
        loss: torch.Tensor = self.loss_fn(preds, batch.target)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch.input.shape[0])
        if isinstance(self.train_metrics, MetricCollection):
            self.train_metrics(preds, batch.target)
            self.log_dict(self.train_metrics, on_step=True, on_epoch=False)
        return loss

    def on_train_epoch_end(self) -> None:
        if isinstance(self.train_metrics, MetricCollection):
            self.train_metrics.reset()

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        preds = self(batch.input)
        loss: torch.Tensor = self.loss_fn(preds, batch.target)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, batch_size=batch.input.shape[0])
        if isinstance(self.val_metrics, MetricCollection):
            self.val_metrics.update(preds, batch.target)
            self.log_dict(self.val_metrics, on_step=False, on_epoch=True, batch_size=batch.input.shape[0])
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        preds = self(batch.input)
        if isinstance(self.test_metrics, MetricCollection):
            self.test_metrics.update(preds, batch.target)
            self.log_dict(self.test_metrics, on_step=False, on_epoch=True, batch_size=batch.input.shape[0])
        if self.predictions_sink is not None:
            self._write_predictions(preds, batch)
        return preds

    def predict_step(self, batch: Any, batch_idx: int) -> Any:
        preds = self(batch.input)
        if self.predictions_sink is not None:
            self._write_predictions(preds, batch)
        return preds

    def configure_optimizers(self) -> Any:
        return flow(self.optimizer, params=self.parameters())

    # ---- the four entry points ---------------------------------------------- #
    @entrypoint("fit", role="trainer", primary=True)
    def fit(self) -> None:
        logger.info(f"sonair training: {self.experiment_name} ({self.run_name})")
        if self.train_set is None:
            raise ValueError("LightningClassifier.fit() requires a 'train_set'.")
        train_ds = ensure_sample_dataset(flow(self.train_set))
        val_ds = ensure_sample_dataset(flow(self.val_set)) if self.val_set is not None else None
        test_ds = ensure_sample_dataset(flow(self.test_set)) if self.test_set is not None else None
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
        dataset = ensure_sample_dataset(flow(self.test_set))
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
        dataset = ensure_sample_dataset(flow(source))
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
        try:
            from liquifai.discovery import get_configurable_paths

            return dict(get_configurable_paths(self))
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
        self.train_set = ensure_sample_dataset(flow(self.train_set))
        if self.val_set is not None:
            self.val_set = ensure_sample_dataset(flow(self.val_set))
        if self.test_set is not None:
            self.test_set = ensure_sample_dataset(flow(self.test_set))

        kind = detect_target_kind(self.train_set)
        if kind == "multilabel":
            raise NotImplementedError("multi-label string targets are not yet supported.")

        nc = self._num_classes_without_data()
        want_weights = self.balance_classes and _loss_accepts_weight(self.loss_fn)
        need_fit = kind == "str" and self.label_map is None
        raw_targets: Optional[List[Any]] = None
        if need_fit or want_weights or nc is None:
            raw_targets = self._walk_targets(self.train_set)

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

        if nc is None:
            if self.label_map is not None:
                nc = self.label_map.num_classes
            else:
                assert raw_targets is not None
                highest = max((_target_to_int(t) for t in raw_targets if t is not None), default=-1)
                nc = highest + 1 if highest >= 0 else None
        if not nc:
            raise RuntimeError(
                "could not determine num_classes — set it on the trainer/model or wire a labelled dataset."
            )
        self.num_classes = int(nc)

        if self.balance_classes and self.class_weights is None:
            self._apply_class_weights(self.num_classes, kind, raw_targets)

        self.model = flow(self.model, num_classes=self.num_classes)
        if self.train_metrics is None:
            base = build_metric_collection(self.metrics, num_classes=self.num_classes)
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
        dataset = ensure_sample_dataset(flow(source))
        kind = detect_target_kind(dataset)
        if kind == "multilabel":
            raise NotImplementedError("multi-label string targets are not yet supported.")
        if kind == "str":
            label_map = self.label_map or self._load_pinned_label_map()
            if label_map is None:
                raise RuntimeError("eval dataset has string targets but no label_map and no class_names.json found.")
            self.label_map = label_map
            dataset = encode_flux(dataset, label_map)
        setattr(self, eval_slot, dataset)
        metrics_slot = "val_metrics" if self.task == "evaluate" else "test_metrics"
        prefix = "val/" if self.task == "evaluate" else "test/"
        if getattr(self, metrics_slot) is None:
            base = build_metric_collection(self.metrics, num_classes=self.num_classes)
            setattr(self, metrics_slot, base.clone(prefix=prefix))

    def _walk_targets(self, source: Any) -> List[Any]:
        src = flow(source)
        total = len(src) if hasattr(src, "__len__") else None
        return list(tqdm(iter_targets(src), total=total, desc="sonair: reading targets", unit="sample"))

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

    def _apply_class_weights(self, num_classes: int, kind: _TargetKind, raw_targets: Optional[List[Any]]) -> None:
        if not _loss_accepts_weight(self.loss_fn):
            name: Any = getattr(self.loss_fn, "target", type(self.loss_fn).__name__)
            logger.warning(f"balance_classes=True but the loss ({name}) has no 'weight' arg; training unweighted.")
            self.class_weights = None
            return
        assert raw_targets is not None
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
        weights = torch.where(counts > 0, total / (num_classes * counts), torch.zeros_like(counts))
        self.class_weights = weights.to(torch.float32)
        if isinstance(self.loss_fn, Fluid):
            self.loss_fn = flow(self.loss_fn, weight=self.class_weights)
        else:
            self.loss_fn.weight = self.class_weights  # type: ignore[assignment]
        logger.info(f"Balanced loss: injected inverse-frequency class weights for {num_classes} classes.")

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
        per_sample_meta = None
        meta = getattr(batch, "metadata", None)
        if isinstance(meta, list):
            per_sample_meta = meta
        elif isinstance(meta, dict):
            per_sample_meta = meta.get("per_sample")
        items = (
            preds["predictions"]
            if isinstance(preds, dict) and "predictions" in preds
            else (preds if isinstance(preds, list) else [preds])
        )
        for i in range(len(items)):
            m = per_sample_meta[i] if per_sample_meta and i < len(per_sample_meta) else {}
            self.predictions_sink.write(items[i], m)

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


__all__ = ["ClassifierTask", "LightningClassifier", "detect_target_kind", "encode_flux"]
