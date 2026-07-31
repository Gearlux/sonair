"""``HuggingFaceClassifier`` — the same classification run, driven by ``transformers.Trainer``.

The fifth backend, and the one that swaps the most while writing the least. Keras and
TensorFlow bring their own array libraries; fastai brings its own loop over torch. This brings
somebody else's *whole training program*: ``Trainer`` owns the epoch, the optimizer, the
scheduler, gradient accumulation, mixed precision and distribution (through Accelerate). What
sonair supplies is the data, the run identity, and the artifacts — the things that are the
task's, not the engine's.

So what is written here is small and specific:

* **:meth:`HuggingFaceClassifier._to_hf_batch`** — one collated record batch to the
  ``{"pixel_values", "labels"}`` mapping ``Trainer`` feeds a model. The twin of the Lightning
  backend's ``_targets``, the Keras backend's ``_to_arrays`` and the fastai backend's
  ``_to_tuple``, handed to ``Trainer`` as its ``data_collator`` exactly as those are handed to
  a ``DataLoader``. It is the whole of what this backend says about batches.
* **``TrainingArguments`` from the shared knobs** — ``max_epochs`` → ``num_train_epochs``,
  ``batch_size`` → ``per_device_train_batch_size``. One flat config drives this backend and
  every other.
* The task dispatch, and a predict loop that keeps the metadata (below).

**There is no loss set for this backend, and that is a fact about transformers rather than an
omission.** An ``AutoModelForImageClassification`` handed ``labels=`` computes its own loss
inside ``forward`` and returns it on the output object — picking it from ``config.problem_type``
(single- vs multi-label). There is nothing for a config to wire, so there is no
``huggingface_classification.py`` (``docs/architecture.md`` §5: a backend contributes a module
only where it contributes something a config can WIRE). What a config CAN choose — which is
what the sentinel ``metrics`` slot is for — is what gets reported.

Minimal flat YAML::

    runnable: !class:sonair.HuggingFaceClassifier
      model: !lazy:sonair.HuggingFaceModel { model_name: google/vit-base-patch16-224 }
    train_set: !class:recordstream.Stream { source: ..., ops: !ref:preprocess }
    batch_size: 32
    max_epochs: 3
"""

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Union, cast

import numpy as np
import torch
from confluid import Fluid, Lazy, Mandatory, configurable, flow, output
from loggair import get_logger
from marainer.artifacts import load_class_names, save_class_names, write_metrics_sidecar
from marainer.run_names import generate_run_name
from marainer.tracking.base import Tracker
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
    is_class_id,
    iter_key,
    multi_hot,
)
from recordstream.runnable import ProgressReporting, TorchRunner, entrypoint, run_entrypoint
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

logger = get_logger(__name__)

#: Which entry point :meth:`HuggingFaceClassifier.run` dispatches to — the same vocabulary
#: every other backend uses, because the verb is a property of the TASK, not of the engine.
ClassifierTask = Literal["fit", "evaluate", "test", "predict"]

#: What the ``metrics`` slot accepts: a name resolved below, or a callable over
#: ``(logits, labels)`` returning a float. Spelled out rather than left as ``Any`` so
#: ``to_pydantic`` (form specs / MCP schemas / editor widgets) has a real type to build from.
HuggingFaceMetric = Union[str, Callable[[np.ndarray, np.ndarray], float]]

__all__ = ["ClassifierTask", "HuggingFaceClassifier"]


def _first_target(source: Any, target_key: str) -> Any:
    """The first non-``None`` target in ``source`` — one peek is enough to know the label kind."""
    for target in iter_key(source, target_key):
        if target is not None:
            return target
    return None


def _run_prefix(task: str) -> str:
    return {"fit": "train", "predict": "predict"}.get(task, "evaluate")


def _accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    """Top-1 accuracy for single-label targets (int64 ids)."""
    return float((logits.argmax(axis=-1) == labels).mean())


def _accuracy_multi(logits: np.ndarray, labels: np.ndarray, threshold: float = 0.0) -> float:
    """Element-wise accuracy for a multi-label target, thresholded in LOGIT space.

    ``0.0`` rather than ``0.5`` because the head emits raw logits, exactly as the Keras
    backend's thresholded metrics take ``thresholds: 0.0``.
    """
    return float(((logits > threshold).astype(labels.dtype) == labels).mean())


#: Names the ``metrics`` slot resolves. Kept a small dict rather than a registry because these
#: are plain FUNCTIONS over numpy — there is no class for confluid to tag, the same situation
#: the fastai backend's `fastai.metrics` names are in.
_METRICS: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "accuracy": _accuracy,
    "accuracy_multi": _accuracy_multi,
}


@configurable(task="classification", role="trainer", framework="huggingface")
class HuggingFaceClassifier(TorchRunner, ProgressReporting):
    """Train, evaluate or predict an image classifier with ``transformers.Trainer``.

    Args:
        model: The classification model — wire ``!lazy:sonair.HuggingFaceModel`` so
            ``initialize()`` can inject the dataset-derived ``num_classes``.
        train_set: Training source (``task="fit"``).
        val_set: Validation source. ``Trainer`` evaluates against it each epoch when wired.
        test_set: Evaluation / prediction source.
        num_classes: Class count. Omitted, it is derived from the data or the label map.
        task: Which entry point ``run()`` dispatches to.
        max_epochs: Training epochs (``TrainingArguments.num_train_epochs``).
        batch_size: Rows per batch (``per_device_train_batch_size``).
        num_workers: DataLoader worker processes (``dataloader_num_workers``).
        logger: ``transformers.TrainerCallback`` instances — the Trainer analogue of the
            Lightning backend's ``logger`` slot, so the same flat key drives both.
        experiment_name: Run identity, recorded in artifacts and tracking.
        run_name: Run identity; a random ``adjective_noun`` when omitted.
        input_key: Record key holding the image.
        target_key: Record key holding the label.
        label_map: Pin the name-to-id encoding instead of fitting it from the train targets.
        learning_rate: Peak LR. The default is transformers' own fine-tuning default, which is
            two orders of magnitude below a from-scratch LR because this backend's whole point
            is starting from pretrained weights.
        weight_decay: ``TrainingArguments.weight_decay``.
        output_dir: Where ``Trainer`` writes checkpoints and its state. Defaults beside the
            metrics sidecar when one is configured, else ``./hf_runs/<run_name>``.
        report_to: Trainer's own experiment-tracking integrations. Defaults to ``"none"`` ON
            PURPOSE: Trainer auto-enables every tracker it can import, so on a workspace that
            has wandb and mlflow installed an innocent run would start reporting to both.
            Tracking here goes through the `trackers` slot, like every other backend.
        metrics: Metric names (``"accuracy"``, ``"accuracy_multi"``) or callables over
            ``(logits, labels)``. ``None`` picks by target shape.
    """

    def __init__(
        self,
        model: Mandatory[Lazy[Any]],
        train_set: Optional[Lazy[RecordSource]] = None,
        val_set: Optional[Lazy[RecordSource]] = None,
        test_set: Optional[Lazy[RecordSource]] = None,
        num_classes: Optional[int] = None,
        task: ClassifierTask = "fit",
        max_epochs: int = 10,
        batch_size: int = 32,
        num_workers: int = 0,
        logger: Optional[Iterable[Any]] = None,
        experiment_name: str = "sonair_classification",
        run_name: Optional[str] = None,
        input_key: str = "image",
        target_key: str = "class",
        label_map: Optional[LabelMap] = None,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.0,
        output_dir: Optional[str] = None,
        report_to: str = "none",
        metrics: Optional[Sequence[HuggingFaceMetric]] = None,
    ) -> None:
        super().__init__()
        # `torch.nn.Module`, not `Lazy[Any]`: the deferred-ness is declared once, on the ctor
        # param, which is what `lazy_param_names` / `to_pydantic` read. A union here would only
        # stop every method calling through the slot.
        self.model: Any = model
        self.train_set: Optional[Lazy[RecordSource]] = train_set
        self.val_set: Optional[Lazy[RecordSource]] = val_set
        self.test_set: Optional[Lazy[RecordSource]] = test_set
        self.task: ClassifierTask = task
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.callbacks: List[Any] = list(logger or [])
        self.experiment_name = experiment_name
        self.run_name = run_name if run_name is not None else generate_run_name(prefix=_run_prefix(task))
        self.input_key = str(input_key)
        self.target_key = str(target_key)
        self.label_map = label_map
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.output_dir = output_dir
        self.report_to = report_to
        # Sentinel, for the same reason every other backend uses one: the right metric depends
        # on whether the targets are single- or multi-label, which is only known once
        # `initialize()` has peeked the data. A live default here could not be told apart from
        # a user's explicit choice.
        self.metrics: Optional[Sequence[HuggingFaceMetric]] = metrics
        self.multilabel: bool = False
        self.predictions_sink: Optional[Lazy[PredictionsSink]] = None
        self.metrics_sidecar_path: Optional[str] = None
        self.trackers: Optional[List[Tracker]] = None
        self._class_names: Optional[List[str]] = None
        self._computed_metrics: Dict[str, float] = {}

    @property
    def __needs_autograd__(self) -> bool:  # type: ignore[override]  # dynamic per-task override
        """Only ``fit`` optimizes — an evaluate/predict run is happy under inference mode."""
        return self.task == "fit"

    # ---- run dispatch ------------------------------------------------------- #
    def run(self) -> None:
        self.initialize()
        # The `@entrypoint` markers below ARE the dispatch table (recordstream docs/runnable.md).
        run_entrypoint(self, self.task)

    # ---- setup -------------------------------------------------------------- #
    def initialize(self) -> None:
        """Resolve labels and ``num_classes``, then build the model at that width."""
        if self.task == "fit":
            self._initialize_fit()
        else:
            self._initialize_eval()
        self._resolve_task_defaults()
        self.model = flow(self.model, num_classes=self.num_classes)
        # `problem_type` is what an `AutoModelForImageClassification` reads in `forward` to pick
        # BCE-with-logits over cross-entropy. Set here rather than on the model wrapper because
        # only the classifier has peeked the targets — the wrapper never sees the data.
        if self.multilabel:
            self._network().config.problem_type = "multi_label_classification"

    def _initialize_fit(self) -> None:
        self.train_set = ensure_record_dataset(self.train_set)
        if self.val_set is not None:
            self.val_set = ensure_record_dataset(self.val_set)
        if self.test_set is not None:
            self.test_set = ensure_record_dataset(self.test_set)

        first = _first_target(self.train_set, self.target_key)
        self.multilabel = isinstance(first, (list, tuple, set))
        probe = next(iter(first), None) if self.multilabel else first
        targets_are_names = probe is not None and not is_class_id(probe)

        if targets_are_names and self.label_map is None:
            self.label_map = LabelMap.fit(iter_key(self.train_set, self.target_key))
            logger.info(f"Fitted LabelMap with {self.label_map.num_classes} classes from train targets.")
        if targets_are_names and self.label_map is not None:
            self.train_set = self.label_map.encode(self.train_set)
            if self.val_set is not None:
                self.val_set = self.label_map.encode(self.val_set)
            if self.test_set is not None:
                self.test_set = self.label_map.encode(self.test_set)

        if self.num_classes is None:
            if self.label_map is not None:
                self.num_classes = self.label_map.num_classes
            else:
                ids = [i for t in iter_key(self.train_set, self.target_key) for i in LabelMap().to_ids(t)]
                self.num_classes = max(ids) + 1 if ids else None
        if not self.num_classes:
            raise RuntimeError("could not determine num_classes — set it, or wire a labelled dataset.")
        self._class_names = class_names(self.train_set, self.val_set, self.test_set)

    def _initialize_eval(self) -> None:
        self.model = flow(self.model)
        count = getattr(self.model, "num_classes", None)
        if not count:
            raise RuntimeError("could not determine num_classes from the model (it must expose `num_classes`).")
        self.num_classes = int(count)
        if self.test_set is None:
            return
        dataset = ensure_record_dataset(self.test_set)
        first = _first_target(dataset, self.target_key)
        self.multilabel = isinstance(first, (list, tuple, set))
        probe = next(iter(first), None) if self.multilabel else first
        if probe is not None and not is_class_id(probe):
            label_map = self.label_map or load_class_names(getattr(self.model, "checkpoint_path", None))
            if label_map is None:
                raise RuntimeError("eval dataset has string targets but no label_map and no class_names.json found.")
            self.label_map = label_map
            dataset = label_map.encode(dataset)
        self.test_set = dataset
        self._class_names = class_names(dataset)

    def _resolve_task_defaults(self) -> None:
        """Fill the unset (``None``) metrics slot. Idempotent; never overwrites.

        There is no loss to resolve — the model computes its own (module docstring). The metric
        default still has to wait for the data, because a multi-label target needs an
        element-wise, logit-thresholded accuracy rather than an argmax one.
        """
        if self.metrics is None:
            self.metrics = ["accuracy_multi"] if self.multilabel else ["accuracy"]

    def _network(self) -> Any:
        """The live ``transformers`` model — the wrapper's, or a bare one wired directly."""
        return getattr(self.model, "model", self.model)

    # ---- data --------------------------------------------------------------- #
    def _to_hf_batch(self, records: Sequence[Any]) -> Dict[str, torch.Tensor]:
        """One list of records → the mapping ``Trainer`` feeds the model.

        Handed to ``Trainer`` as its ``data_collator``, which is the same seat a ``collate_fn``
        occupies in the other torch backends. ``collate_records`` stacks the already-tensorized
        image to ``[N, 3, H, W]``; the keys are transformers' contract, not ours —
        ``pixel_values`` is what an image model's ``forward`` takes and ``labels`` is what makes
        it return a loss.

        int64 is named explicitly for the same reason as every other backend: a dataset yielding
        int32 label tensors is legal and the loss rejects it. A multi-label target instead needs
        FLOAT multi-hot, because ``problem_type="multi_label_classification"`` routes to
        ``BCEWithLogitsLoss``, which compares against float targets.
        """
        batch = collate_records(list(records))
        pixel_values = batch_tensor(batch, self.input_key)
        if self.multilabel:
            if not self.num_classes:
                raise RuntimeError("HuggingFaceClassifier: a multi-label target needs num_classes.")
            labels = torch.as_tensor(multi_hot(batch, self.target_key, self.num_classes), dtype=torch.float32)
        else:
            labels = batch_tensor(batch, self.target_key, dtype=torch.int64)
        return {"pixel_values": pixel_values, "labels": labels}

    def _record_loader(self, source: Any) -> DataLoader:
        """A loader of COLLATED RECORD BATCHES — the shape :meth:`predict` needs.

        Prediction pairs each output row with the metadata that travelled beside the image, and
        :meth:`_to_hf_batch` has already dropped it. So the predict path keeps the record dict
        and reads it with recordstream's generic readers, exactly as the other backends do.
        """
        return DataLoader(
            # `DataLoader` is duck-typed at RUN time (it needs `__getitem__`/`__len__`, which a
            # map-style `Stream` has) but its stubs demand the nominal torch base, which a
            # `Stream` deliberately does not inherit — that would put torch in recordstream's core.
            cast("TorchDataset[Any]", ensure_record_dataset(source)),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_records,
        )

    # ---- the Trainer -------------------------------------------------------- #
    def _training_arguments(self) -> Any:
        """``TrainingArguments`` from the shared knobs — the flat config, in Trainer's words."""
        from transformers import TrainingArguments

        return TrainingArguments(
            output_dir=self._output_dir(),
            num_train_epochs=self.max_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            dataloader_num_workers=self.num_workers,
            # A generic map-style source is not a `datasets.Dataset`, so Trainer must not try to
            # drop "unused" columns by inspecting the model signature — our collate already
            # produces exactly the two keys the model takes.
            remove_unused_columns=False,
            report_to=self.report_to,
            run_name=self.run_name,
            disable_tqdm=True,
            save_strategy="no",
        )

    def _build_trainer(self, train: Any = None, evaluate: Any = None) -> Any:
        from transformers import Trainer

        return Trainer(
            model=self._network(),
            args=self._training_arguments(),
            train_dataset=train,
            eval_dataset=evaluate,
            data_collator=self._to_hf_batch,
            compute_metrics=self._compute_metrics,
            callbacks=[flow(cb) for cb in self.callbacks] or None,
        )

    def _compute_metrics(self, eval_pred: Any) -> Dict[str, float]:
        """Trainer's metric hook, wearing THIS task's metric choice."""
        logits, labels = eval_pred.predictions, eval_pred.label_ids
        logits = logits[0] if isinstance(logits, tuple) else logits
        out: Dict[str, float] = {}
        for metric in self.metrics or []:
            fn = _as_metric(metric)
            out[getattr(fn, "__name__", str(metric)).lstrip("_")] = float(fn(np.asarray(logits), np.asarray(labels)))
        return out

    def _output_dir(self) -> str:
        """Where Trainer writes its state — beside the sidecar when one is configured."""
        if self.output_dir:
            return str(Path(self.output_dir).expanduser())
        if self.metrics_sidecar_path:
            return str(Path(self.metrics_sidecar_path).expanduser().parent / self.run_name)
        return str(Path("./hf_runs") / self.run_name)

    # ---- the four entry points ---------------------------------------------- #
    @entrypoint("fit", role="trainer", primary=True)
    def fit(self) -> None:
        logger.info(f"sonair huggingface training: {self.experiment_name} ({self.run_name})")
        if self.train_set is None:
            raise ValueError("HuggingFaceClassifier.fit() requires a 'train_set'.")
        trainer = self._build_trainer(train=self.train_set, evaluate=self.val_set)
        result = trainer.train()
        self._computed_metrics = {k: float(v) for k, v in result.metrics.items() if isinstance(v, (int, float))}
        if self.val_set is not None:
            self._computed_metrics.update({k: float(v) for k, v in trainer.evaluate().items() if _is_number(v)})
        logger.info(f"Training complete: {self._computed_metrics}")
        save_class_names(self._class_names, self._artifact_dir(), self._live_trackers())
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)

    @entrypoint("evaluate", role="evaluator")
    def evaluate(self) -> None:
        self.test()

    @entrypoint("test", role="evaluator", primary=True)
    def test(self) -> None:
        if self.test_set is None:
            raise ValueError("HuggingFaceClassifier.test() requires a 'test_set'.")
        results = self._build_trainer(evaluate=self.test_set).evaluate()
        self._computed_metrics = {k: float(v) for k, v in results.items() if _is_number(v)}
        logger.info(f"Evaluation complete: {self._computed_metrics}")
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)

    @entrypoint("predict", role="predictor", primary=True)
    def predict(self) -> None:
        """Stream per-record predictions to the sink.

        A plain ``torch.no_grad()`` forward rather than ``Trainer.predict``: that returns only
        the stacked logits, and the sink contract needs each row paired with the metadata from
        the record it came from. One walk over the record loader keeps the two in step by
        construction — the same choice the fastai backend makes for the same reason.
        """
        if self.test_set is None:
            raise ValueError("HuggingFaceClassifier.predict() requires a 'test_set'.")
        sink = flow(self.predictions_sink)
        network = self._network().eval()
        device = next(network.parameters()).device
        with torch.no_grad():
            for batch in self._record_loader(self.test_set):
                logits = network(pixel_values=batch_tensor(batch, self.input_key, device)).logits
                if sink is not None:
                    self._write_predictions(logits, batch, cast(PredictionsSink, sink))
        if sink is not None and hasattr(sink, "close"):
            sink.close()
        logger.info("Prediction complete.")

    # ---- predictions -------------------------------------------------------- #
    def _write_predictions(self, logits: torch.Tensor, batch: Any, sink: PredictionsSink) -> None:
        """One :class:`~recordstream.ClassificationOutput` per record, paired with its metadata.

        ``classification_output`` builds the CONTRACT (logits / probs / class_idx) — a sink reads
        those by name and cannot softmax on the caller's behalf, since only the model knows
        whether its head already did. The contract is generic in its array type, which is why the
        very same sink consumes this output and the Keras backend's numpy one.
        """
        batched = classification_output(logits)
        per_record_meta = batch_metadata(batch, exclude=(self.input_key, self.target_key))
        for i in range(len(logits)):
            row: ClassificationOutput[torch.Tensor] = {
                "logits": batched["logits"][i],
                "probs": batched["probs"][i],
                "class_idx": batched["class_idx"][i],
            }
            meta = per_record_meta[i] if per_record_meta and i < len(per_record_meta) else {}
            sink.write(row, meta)

    # ---- artifacts + tracking ----------------------------------------------- #
    def _artifact_dir(self) -> Optional[str]:
        """Where run artifacts land. Trainer's own output dir is the natural home."""
        if self.metrics_sidecar_path:
            return str(Path(self.metrics_sidecar_path).expanduser().parent)
        return self._output_dir()

    def _live_trackers(self) -> List[Tracker]:
        return [flow(t) for t in (self.trackers or [])]

    # ---- @outputs ------------------------------------------------------------ #
    @property
    @output
    def trained_model(self) -> Any:
        """The live trained model (post-run ``@output``). Raises while still deferred."""
        if isinstance(self.model, Fluid):
            raise RuntimeError("trained_model is only available after run() — the model is still deferred.")
        return self.model

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


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_metric(value: HuggingFaceMetric) -> Callable[[np.ndarray, np.ndarray], float]:
    """A metric callable from a callable or from its NAME.

    The string form is not sugar: these are plain FUNCTIONS over numpy, so there is no class
    for a ``!class:`` tag to name or for confluid's registry to tag — the same situation the
    fastai backend's ``fastai.metrics`` names are in.
    """
    built = flow(value)
    if callable(built):
        return cast(Callable[[np.ndarray, np.ndarray], float], built)
    metric = _METRICS.get(str(built))
    if metric is None:
        raise ValueError(f"Unknown metric {built!r}. Known names: {', '.join(sorted(_METRICS))}.")
    return metric
