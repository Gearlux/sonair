"""``MlxClassifier`` — the same classification run on Apple's MLX.

Since P6 (2026-08-08) this is ``(ClassificationBase, matrainer.mlx.MlxLoopRunnable)``: the
loop machinery — the ``nn.value_and_grad`` step with the per-step ``mx.eval``, the train/eval
MODE flips at the loop boundaries, the epoch driver with the per-epoch reshuffle, the
plain-function tracking hookup, the record-batch predict walk — lives in
:class:`matrainer.mlx.loop.MlxLoopRunnable`; the label pipeline in
:class:`~sonair.classification_base.ClassificationBase`. What stays HERE is classification's
mlx vocabulary:

* **Targets are int32**, the one dtype in the fleet that differs from the int64 every other
  backend names: MLX indexes with int32 and its ``cross_entropy`` expects it.
* **``binary_cross_entropy`` is bound ``with_logits=True``** for a multi-label run — the head
  emits raw logits and the MLX default expects probabilities: the same silent failure the
  Keras string losses have, a finite loss that simply does not train.
* **``f1_macro`` is accumulated over the split** (one confusion matrix, reduced once) — a
  per-batch average reads three to five points LOW; a multi-label run DROPS a confusion
  metric with a warning rather than approximating one.
* **The class names persist in :meth:`_fit_epilogue`** — beside the sidecar, onto the trackers.

**Apple silicon ONLY** — the extra's platform marker keeps ``pip install sonair[mlx]``
succeeding (and empty) elsewhere.
"""

from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union, cast

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from confluid import Lazy, Mandatory, configurable, flow
from loggair import get_logger
from matrainer.artifacts import save_class_names
from matrainer.metric_functions import accumulate_confusion, macro_f1, zero_confusion
from matrainer.metric_names import MetricNames
from matrainer.metric_slots import merge_metric_specs
from matrainer.mlx.loop import EpochCallback, MlxLoopRunnable, MlxLoss, MlxMetric, MlxOptimizer
from matrainer.tracking.base import Tracker
from recordstream import ClassificationOutput, LabelMap, RecordSource, batch_values, multi_hot

from sonair.classification_base import ClassificationBase

logger = get_logger(__name__)

#: Which entry point :meth:`MlxClassifier.run` dispatches to.
ClassifierTask = Literal["fit", "evaluate", "test", "predict"]

__all__ = ["ClassifierTask", "EpochCallback", "MlxClassifier"]


def _accuracy(logits: Any, labels: Any) -> Any:
    """Top-1 accuracy for single-label targets (int32 ids)."""
    return mx.mean(mx.argmax(logits, axis=-1) == labels)


def _accuracy_multi(logits: Any, labels: Any) -> Any:
    """Element-wise accuracy for a multi-label target, thresholded in LOGIT space at 0.0."""
    return mx.mean((logits > 0.0).astype(labels.dtype) == labels)


#: Metric names this backend resolves PER BATCH. Plain functions — nothing for confluid to tag.
_METRICS: Dict[str, Callable[..., Any]] = {"accuracy": _accuracy, "accuracy_multi": _accuracy_multi}

#: Metric names resolved from the split's CONFUSION MATRIX rather than per batch — averaging
#: them over batches gives a systematically worse number (``matrainer.metric_functions``).
#: Single-label only: a multi-hot row has no single cell to land in.
_CONFUSION_METRICS: Dict[str, Callable[[Any], float]] = {"f1_macro": macro_f1}


@configurable(task="classification", role="trainer", framework="mlx")
class MlxClassifier(ClassificationBase, MlxLoopRunnable):
    """Train, evaluate or predict an image classifier with MLX.

    Args:
        model: The classification model — wire ``!lazy:sonair.MlxConvNet`` so ``initialize()``
            can inject the dataset-derived ``num_classes``.
        train_set: Training source (``task="fit"``).
        val_set: Validation source, scored at the end of each epoch when wired.
        test_set: Evaluation / prediction source.
        num_classes: Class count. Omitted, it is derived from the data or the label map.
        task: Which entry point ``run()`` dispatches to.
        max_epochs: Training epochs.
        batch_size: Rows per batch.
        trackers: matrainer trackers — one or a list.
        callbacks: Callables invoked as ``callback(epoch, logs)`` after each epoch.
        experiment_name / run_name: tracking identity.
        input_key / target_key: Record keys for the image and the label.
        label_map: Pin the name-to-id encoding instead of fitting it from the train targets.
        learning_rate: Learning rate for the optimizer.
        optimizer: An ``mlx.optimizers`` CLASS name (``"Adam"``, ``"AdamW"``, ``"SGD"``).
        loss: An ``mlx.nn.losses`` name or callable. ``None`` picks by target shape.
        metrics: Metric names (``"accuracy"``, ``"accuracy_multi"``, ``"f1_macro"``) or
            callables. ``None`` picks by target shape.
        train_metrics / val_metrics / test_metrics: That split's OWN metrics, merged over
            ``metrics``.
        metric_names: How recorded metrics are named — this loop already writes the canonical
            ``<split>/<name>``, so the default translates nothing.
    """

    def __init__(
        self,
        model: Mandatory[Lazy[nn.Module]],
        train_set: Optional[Lazy[RecordSource]] = None,
        val_set: Optional[Lazy[RecordSource]] = None,
        test_set: Optional[Lazy[RecordSource]] = None,
        num_classes: Optional[int] = None,
        task: ClassifierTask = "fit",
        max_epochs: int = 10,
        batch_size: int = 32,
        trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = None,
        callbacks: Optional[List[Lazy[EpochCallback]]] = None,
        experiment_name: str = "sonair_classification",
        run_name: Optional[str] = None,
        input_key: str = "image",
        target_key: str = "class",
        label_map: Optional[LabelMap] = None,
        learning_rate: float = 1e-3,
        optimizer: MlxOptimizer = "Adam",
        loss: Optional[MlxLoss] = None,
        metrics: Optional[Sequence[MlxMetric]] = None,
        train_metrics: Optional[Sequence[MlxMetric]] = None,
        val_metrics: Optional[Sequence[MlxMetric]] = None,
        test_metrics: Optional[Sequence[MlxMetric]] = None,
        metric_names: Optional[Lazy[MetricNames]] = None,
    ) -> None:
        super().__init__(
            model=model,
            train_set=train_set,
            test_set=test_set,
            task=task,
            max_epochs=max_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            trackers=trackers,
            callbacks=callbacks,
            experiment_name=experiment_name,
            run_name=run_name,
            input_key=input_key,
            target_key=target_key,
            metric_names=metric_names,
        )
        self.val_set = val_set
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.label_map = label_map
        self.multilabel: bool = False
        # Balancing is a loss-side convention the MLX functions do not carry — the task base's
        # capability probe answers no.
        self.balance_classes = False
        self._class_names: Optional[List[str]] = None

    # ---- the task hooks ---------------------------------------------------------------------- #
    def _default_mlx_loss(self) -> Optional[MlxLoss]:
        return "binary_cross_entropy" if self.multilabel else "cross_entropy"

    def _default_mlx_metrics(self) -> Optional[List[Any]]:
        return ["accuracy_multi"] if self.multilabel else ["accuracy"]

    def _loss_fn(self) -> Callable[..., Any]:
        """The resolved loss, meaned — with the multi-label ``with_logits`` binding."""
        chosen = self.loss if self.loss is not None else self._default_mlx_loss()
        loss = _as_loss(chosen, self.multilabel)

        def compute(outputs: Any, targets: Any) -> Any:
            return mx.mean(loss(outputs, targets))

        return compute

    def _to_arrays(self, batch: Any, with_targets: bool = True) -> Any:
        """float32 image; ``multi_hot`` for a multi-label target, else int32 ids.

        int32, not int64: MLX indexes with int32 and its ``cross_entropy`` expects it.
        """
        x = np.asarray(batch_values(batch, self.input_key), dtype="float32")
        if not with_targets:
            return (x,)
        if self.multilabel:
            if not self.num_classes:
                raise RuntimeError("MlxClassifier: a multi-label target needs num_classes.")
            return x, multi_hot(batch, self.target_key, self.num_classes)
        return x, np.asarray(batch_values(batch, self.target_key), dtype="int32")

    # ---- the metric state: batchwise sums + ONE confusion matrix per split ------------------- #
    def _partition_metrics(self, split: str) -> Any:
        batchwise: List[MlxMetric] = []
        confusion: Dict[str, Callable[[Any], float]] = {}
        for metric in merge_metric_specs(self.metrics, getattr(self, f"{split}_metrics")) or []:
            name = flow(metric)
            if isinstance(name, str) and name in _CONFUSION_METRICS:
                confusion[name] = _CONFUSION_METRICS[name]
            else:
                batchwise.append(metric)
        return batchwise, confusion

    def _split_confusion(self, confusion: Dict[str, Callable[[Any], float]]) -> Optional[np.ndarray]:
        """A zeroed accumulator when this split wants one; a multi-label run DROPS it loudly."""
        if not confusion or self.multilabel:
            if confusion and self.multilabel:
                logger.warning(
                    f"MlxClassifier: {', '.join(sorted(confusion))} needs single-label targets and this run is "
                    f"multi-label — the metric is not reported. Use `accuracy_multi` for a multi-hot target."
                )
            return None
        if not self.num_classes:
            raise RuntimeError("MlxClassifier: a confusion-matrix metric needs num_classes.")
        return zero_confusion(self.num_classes)

    def _new_split_state(self, split: str) -> Any:
        batchwise, confusion = self._partition_metrics(split)
        return {
            "batchwise": batchwise,
            "confusion": confusion,
            "matrix": self._split_confusion(confusion),
            "sums": {},
            "batches": 0,
        }

    def _score_batch(self, state: Any, outputs: Any, arrays: Tuple[Any, ...], split: str) -> None:
        labels = arrays[1]
        state["batches"] += 1
        for metric in state["batchwise"]:
            fn = _as_metric(metric)
            name = getattr(fn, "__name__", str(metric)).lstrip("_")
            state["sums"][name] = state["sums"].get(name, 0.0) + float(fn(outputs, labels))
        if state["matrix"] is not None:
            accumulate_confusion(state["matrix"], np.asarray(outputs), np.asarray(labels))

    def _split_scores(self, state: Any, split: str, step: Optional[int] = None) -> Dict[str, float]:
        if not state["batches"]:
            return {}
        out = {name: value / state["batches"] for name, value in state["sums"].items()}
        if state["matrix"] is not None:
            out.update({name: fn(state["matrix"]) for name, fn in state["confusion"].items()})
        return out

    # ---- artifacts ---------------------------------------------------------------------------- #
    def _fit_epilogue(self) -> None:
        """Persist the run's label vocabulary beside the sidecar and onto the trackers."""
        save_class_names(self._class_names, self._artifact_dir(), self._live_trackers())

    def _artifact_dir(self) -> Optional[str]:
        if self.metrics_sidecar_path:
            return str(Path(self.metrics_sidecar_path).expanduser().parent)
        return None

    # ---- predictions -------------------------------------------------------------------------- #
    def _prediction_output(self, preds: Any) -> Any:
        """The batched :class:`~recordstream.ClassificationOutput` — the machinery slices it."""
        logits = np.asarray(preds)
        out: ClassificationOutput[np.ndarray] = {
            "logits": logits,
            "probs": _softmax(logits),
            "class_idx": np.argmax(logits, axis=-1),
        }
        return out


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax. The model emits raw logits (no head activation), like every twin."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return cast(np.ndarray, exp / exp.sum(axis=-1, keepdims=True))


def _as_loss(value: Optional[MlxLoss], multilabel: bool = False) -> Callable[..., Any]:
    """An MLX loss from a callable or its NAME in ``mlx.nn.losses``.

    ``binary_cross_entropy`` is bound with ``with_logits=True`` for the multi-label case,
    because the head emits raw logits and the MLX default expects probabilities. That is the
    same trap the Keras backend documents for its string losses, and it fails the same silent
    way: a finite loss that simply does not train.
    """
    built = flow(value)
    if callable(built):
        return cast(Callable[..., Any], built)
    name = str(built)
    loss = getattr(nn.losses, name, None)
    if loss is None:
        raise ValueError(
            f"Unknown MLX loss {name!r}. Names come from `mlx.nn.losses` — e.g. cross_entropy, "
            "binary_cross_entropy, hinge_loss."
        )
    if multilabel and name == "binary_cross_entropy":
        return cast(Callable[..., Any], partial(loss, with_logits=True))
    return cast(Callable[..., Any], loss)


def _as_metric(value: MlxMetric) -> Callable[..., Any]:
    """A PER-BATCH metric from a callable or its NAME (``"accuracy"`` / ``"accuracy_multi"``).

    A confusion-matrix name (``"f1_macro"``) never reaches here — ``_partition_metrics`` takes
    those out first — but it is listed in the error, because "unknown metric" for a name the
    backend does support would send the reader looking in the wrong place.
    """
    built = flow(value)
    if callable(built):
        return cast(Callable[..., Any], built)
    metric = _METRICS.get(str(built))
    if metric is None:
        known = sorted({*_METRICS, *_CONFUSION_METRICS})
        raise ValueError(f"Unknown metric {built!r}. Known names: {', '.join(known)}.")
    return metric
