"""``HuggingFaceClassifier`` — the same classification run, driven by ``transformers.Trainer``.

Since P6 (2026-08-08) this is ``(ClassificationBase, matrainer.huggingface.
HuggingFaceRunnable)``: the machinery — the deferred ``TrainingArguments`` / ``Trainer``
slots, the run-time argument FLOOR, the entry points with the plain-function tracking hookup,
the ``log_history`` replay, the record-loader predict walk — lives in
:class:`matrainer.huggingface.trainer.HuggingFaceRunnable`; the label pipeline (the
multilabel peek, ``LabelMap`` fitting/encoding, ``num_classes`` resolution) in
:class:`~sonair.classification_base.ClassificationBase`. What stays HERE is classification's
huggingface vocabulary:

* **:meth:`_to_hf_batch`** — one collated record batch to ``{"pixel_values", "labels"}``:
  int64 ids for a single-label target, FLOAT multi-hot for a multi-label one, because
  ``problem_type="multi_label_classification"`` routes to ``BCEWithLogitsLoss``, which
  compares against float targets.
* **``initialize()`` sets ``config.problem_type``** on the built model when the peek says
  multi-label — on the CLASSIFIER, because only it has peeked the targets; the model wrapper
  never sees the data.
* **The metric names** (``accuracy`` / ``accuracy_multi`` / ``f1_macro``) — plain functions
  over ``(logits, labels)`` numpy arrays, resolved by :meth:`_metrics_hook`. There is no
  class for a ``!class:`` tag to name, the same situation the fastai names are in.
* **The class names persist in :meth:`_after_fit`** — beside the sidecar, onto the trackers.

**There is no loss set for this backend, and that is a fact about transformers rather than an
omission.** An ``AutoModelForImageClassification`` handed ``labels=`` computes its own loss
inside ``forward``, picking it from ``config.problem_type`` — so there is nothing a config
could wire and therefore no ``huggingface_classification.py``.

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
from confluid import Lazy, Mandatory, configurable, flow
from loggair import get_logger
from matrainer.artifacts import save_class_names
from matrainer.huggingface.trainer import HuggingFaceCallback, HuggingFaceRunnable
from matrainer.metric_functions import accumulate_confusion, macro_f1, zero_confusion
from matrainer.metric_names import MetricNames
from matrainer.metric_slots import merge_metric_specs
from matrainer.tracking.base import Tracker
from recordstream import (
    LabelMap,
    RecordSource,
    batch_tensor,
    class_names,
    classification_output,
    collate_records,
    multi_hot,
)

from sonair.classification_base import ClassificationBase
from sonair.huggingface_model import HuggingFaceModel

logger = get_logger(__name__)

#: Which entry point :meth:`HuggingFaceClassifier.run` dispatches to — the same vocabulary
#: every other backend uses, because the verb is a property of the TASK, not of the engine.
ClassifierTask = Literal["fit", "evaluate", "test", "predict"]

#: What the ``metrics`` slot accepts: a name resolved below, or a callable over
#: ``(logits, labels)`` returning a float. Spelled out rather than left as ``Any`` so
#: ``to_pydantic`` (form specs / MCP schemas / editor widgets) has a real type to build from.
HuggingFaceMetric = Union[str, Callable[[np.ndarray, np.ndarray], float]]

#: What the ``model`` slot admits: the :class:`~sonair.huggingface_model.HuggingFaceModel`
#: wrapper (the usual wiring — it builds the checkpoint at ``flow()`` time with the
#: dataset-derived width), or a live ``transformers`` model handed in directly, which
#: ``_network()`` unwraps to either way. The second arm is ``torch.nn.Module`` rather than
#: ``transformers.PreTrainedModel`` because every ``AutoModelFor*`` IS one, and naming the
#: transformers class would force a module-level ``import transformers`` this module
#: deliberately defers to the machinery's method bodies.
HuggingFaceNetwork = Union[HuggingFaceModel, torch.nn.Module]

__all__ = ["ClassifierTask", "HuggingFaceCallback", "HuggingFaceClassifier", "HuggingFaceNetwork"]


def _accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    """Top-1 accuracy for single-label targets (int64 ids)."""
    return float((logits.argmax(axis=-1) == labels).mean())


def _accuracy_multi(logits: np.ndarray, labels: np.ndarray, threshold: float = 0.0) -> float:
    """Element-wise accuracy for a multi-label target, thresholded in LOGIT space.

    ``0.0`` rather than ``0.5`` because the head emits raw logits, exactly as the Keras
    backend's thresholded metrics take ``thresholds: 0.0``.
    """
    return float(((logits > threshold).astype(labels.dtype) == labels).mean())


def _f1_macro(logits: np.ndarray, labels: np.ndarray) -> float:
    """Macro-averaged F1 for single-label targets (int64 ids).

    A plain ``(logits, labels)`` function like its neighbours, and that is a property of WHERE
    it runs: ``compute_metrics`` is handed the WHOLE split at once, so the confusion matrix is
    built in one call. The JAX and MLX backends need the same arithmetic split across an epoch
    of batches, which is why the counting and the reduction live in
    :mod:`matrainer.metric_functions` rather than being written out here — one implementation,
    three engines, and a number comparable with the torch backends' torchmetrics one.
    """
    matrix = zero_confusion(int(logits.shape[-1]))
    accumulate_confusion(matrix, logits, labels)
    return macro_f1(matrix)


#: Names the ``metrics`` slot resolves. Kept a small dict rather than a registry because these
#: are plain FUNCTIONS over numpy — there is no class for confluid to tag, the same situation
#: the fastai backend's `fastai.metrics` names are in.
_METRICS: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "accuracy": _accuracy,
    "accuracy_multi": _accuracy_multi,
    "f1_macro": _f1_macro,
}


@configurable(task="classification", role="trainer", framework="huggingface")
class HuggingFaceClassifier(ClassificationBase, HuggingFaceRunnable):
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
        trackers: matrainer trackers — one or a list. This run's ONLY output wiring:
            metrics, provenance and the run manifest all go to them.
        callbacks: ``transformers.TrainerCallback`` instances, handed to the ``Trainer``.
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
        metrics: Metric names (``"accuracy"``, ``"accuracy_multi"``, ``"f1_macro"``) or callables over
            ``(logits, labels)``. ``None`` picks by target shape.
        val_metrics / test_metrics: That split's OWN metrics, merged over ``metrics``. There is
            no ``train_metrics``: ``compute_metrics`` is the Trainer's EVALUATION hook.
        metric_names: How recorded metrics are named — see
            :class:`~matrainer.metric_names.MetricNames`. A ``Trainer`` writes ``train_*`` and
            ``eval_*``; the default translates those to the canonical ``train/*`` and ``val/*``
            on the way to the trackers and the sidecar.

    Post-construction slots (YAML / broadcasting): ``training_arguments`` (every
    ``transformers.TrainingArguments`` knob the ctor does not restate — gradient accumulation,
    mixed precision, warmup, scheduler, …; the derived knobs are a run-time FLOOR, so an
    override ADDS to them), ``trainer`` (the ``transformers.Trainer`` itself: ``optimizers``,
    ``processing_class``, a Trainer SUBCLASS), ``test_loader`` (the record-dict ``DataLoader``
    :meth:`predict` walks), ``predictions_sink``, ``metrics_sidecar_path``.
    """

    def __init__(
        self,
        model: Mandatory[Lazy[HuggingFaceNetwork]],
        train_set: Optional[Lazy[RecordSource]] = None,
        val_set: Optional[Lazy[RecordSource]] = None,
        test_set: Optional[Lazy[RecordSource]] = None,
        num_classes: Optional[int] = None,
        task: ClassifierTask = "fit",
        max_epochs: int = 10,
        batch_size: int = 32,
        num_workers: int = 0,
        trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = None,
        callbacks: Optional[List[HuggingFaceCallback]] = None,
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
        val_metrics: Optional[Sequence[HuggingFaceMetric]] = None,
        test_metrics: Optional[Sequence[HuggingFaceMetric]] = None,
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
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            report_to=report_to,
            output_dir=output_dir,
            trackers=trackers,
            callbacks=callbacks,
            experiment_name=experiment_name,
            run_name=run_name,
            input_key=input_key,
            target_key=target_key,
            metrics=metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            metric_names=metric_names,
        )
        self.val_set = val_set
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.label_map = label_map
        self.multilabel: bool = False
        # Balancing is a loss-side convention and this backend has no loss slot — the model
        # computes its own from `problem_type` — so the base's capability probe answers no.
        self.balance_classes = False
        self._class_names: Optional[List[str]] = None

    # ---- setup: the label pipeline is the task base's; the problem_type is ours ------------- #
    def initialize(self) -> None:
        """Resolve labels and ``num_classes`` (task base), then route the loss choice.

        ``problem_type`` is what an ``AutoModelForImageClassification`` reads in ``forward``
        to pick BCE-with-logits over cross-entropy. Set here rather than on the model wrapper
        because only the classifier has peeked the targets — the wrapper never sees the data.
        """
        super().initialize()
        if self.multilabel:
            self._network().config.problem_type = "multi_label_classification"

    def _default_hf_metrics(self) -> Optional[List[Any]]:
        """A multi-label target needs an element-wise, logit-thresholded accuracy."""
        return ["accuracy_multi"] if self.multilabel else ["accuracy"]

    # ---- data ------------------------------------------------------------------------------- #
    def _to_hf_batch(self, records: Sequence[Any]) -> Dict[str, torch.Tensor]:
        """One list of records → the mapping ``Trainer`` feeds the model.

        int64 is named explicitly for the same reason as every other backend: a dataset
        yielding int32 label tensors is legal and the loss rejects it. A multi-label target
        instead needs FLOAT multi-hot, because ``problem_type="multi_label_classification"``
        routes to ``BCEWithLogitsLoss``, which compares against float targets.
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

    # ---- metrics ---------------------------------------------------------------------------- #
    def _metrics_hook(self, split: str) -> Any:
        """Trainer's ``compute_metrics``, bound to ONE split's resolved metric list."""
        metrics = merge_metric_specs(self.metrics, getattr(self, f"{split}_metrics")) or []

        def compute_metrics(eval_pred: Any) -> Dict[str, float]:
            logits, labels = eval_pred.predictions, eval_pred.label_ids
            logits = logits[0] if isinstance(logits, tuple) else logits
            out: Dict[str, float] = {}
            for metric in metrics:
                fn = _as_metric(metric)
                name = getattr(fn, "__name__", str(metric)).lstrip("_")
                out[name] = float(fn(np.asarray(logits), np.asarray(labels)))
            return out

        return compute_metrics

    # ---- artifacts --------------------------------------------------------------------------- #
    def _after_fit(self, trainer: Any, train_ds: Any, val_ds: Any, test_ds: Any) -> None:
        """Persist the run's label vocabulary beside the sidecar and onto the trackers."""
        self._class_names = class_names(train_ds, val_ds, test_ds)
        save_class_names(self._class_names, self._artifact_dir(), self._live_trackers())

    def _artifact_dir(self) -> Optional[str]:
        """Where run artifacts land. Trainer's own output dir is the natural home."""
        if self.metrics_sidecar_path:
            return str(Path(self.metrics_sidecar_path).expanduser().parent)
        return self._output_dir()

    # ---- predictions ------------------------------------------------------------------------- #
    def _prediction_output(self, preds: Any) -> Any:
        """The batched :class:`~recordstream.ClassificationOutput` — the machinery slices it.

        ``classification_output`` builds the CONTRACT (logits / probs / class_idx) — a sink
        reads those by name and cannot softmax on the caller's behalf, since only the model
        knows whether its head already did.
        """
        return classification_output(preds)


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
