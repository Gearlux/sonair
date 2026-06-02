"""LightningClassificationEvaluator: specialised image-classification evaluator.

Subclass of :class:`~marainer.evaluator.LightningEvaluator` that bakes in
classification-specific defaults so a minimal YAML config only needs to specify
the dataset and the (trained) model.

Baked-in defaults (all overridable via YAML ``@configurable`` attributes):

* ``loader`` ``collate_fn`` — :func:`~sonair.classification.classification_collate_fn`
* ``metrics`` — :class:`~torchmetrics.classification.MulticlassAccuracy` (top-1)
  + :class:`~torchmetrics.classification.MulticlassF1Score` (macro).

``num_classes`` is NOT configured here — for evaluation the trained model's head is
authoritative, so it is read off the live model in :meth:`initialize`.

Minimal YAML example::

    evaluator: !class:sonair.evaluator.LightningClassificationEvaluator
      model: !class:sonair.timm_classifier.TimmModel
        model_name: efficientvit_b0
        num_classes: 37            # the model carries the class count
        pretrained: false
        checkpoint_path: ./checkpoints/last.ckpt
      dataset: !class:dataflux.sources.HuggingFaceSource
        path: kitofrank/RFUAV
        split: validation
"""

from pathlib import Path
from typing import Any, Optional, Union

from confluid import Fluid, LazyClass, configurable, flow
from dataflux import LabelMap
from logflow import get_logger
from marainer.evaluator import LightningEvaluator
from marainer.trainer import build_metric_collection, ensure_sample_dataset
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from sonair.classification import classification_collate_fn
from sonair.trainer import detect_target_kind, encode_flux

logger = get_logger(__name__)


@configurable(task="classification", role="evaluator")
class LightningClassificationEvaluator(LightningEvaluator):
    """Specialised image-classification evaluator with baked-in defaults.

    Mirrors :class:`~marainer.evaluator.LightningEvaluator`'s minimal-ctor
    discipline: only ``model`` / ``dataset`` (+ identity ``experiment_name`` /
    ``run_name``); ``num_classes`` is NOT a knob — it is read off the trained
    model in :meth:`initialize`. The classification defaults —
    the ``test/``-prefixed metric set and the ``classification_collate_fn`` loader —
    are deferred (body attributes with `LazyClass` defaults); ``predictions_sink`` /
    ``lightning`` / ``metrics_sidecar_path`` inherit the base defaults and are
    configured post-construction (YAML / broadcasting).

    Metrics are built in :meth:`initialize` (called at the top of :meth:`run`) from the
    ``metrics`` dict of single-metric templates, once ``num_classes`` is resolved.

    Args:
        model: Classification model that emits ``[N, num_classes]`` logits.
            The model is responsible for loading its own weights — pass
            ``checkpoint_path`` to the model class.
        dataset: Evaluation dataset or Flux pipeline.
        experiment_name: MLflow experiment name.
        run_name: MLflow run name. ``None`` (default) auto-generates a random slug like
            ``evaluate_funny_rabbit`` (via the base; see :func:`marainer.run_names.generate_run_name`).
        label_map: Optional :class:`~dataflux.LabelMap` (name→id) pinning the class encoding. When
            the eval dataset's targets are *strings* it is used to encode them to ids (mirroring the
            trainer). When omitted, it is reloaded from the ``class_names.json`` written next to the
            model's checkpoint at train time (via ``LabelMap.load``). Integer targets need no map.
        metrics: Dict of single-metric ``LazyClass`` templates flowed with ``num_classes``
            into a ``test/``-prefixed ``MetricCollection`` in :meth:`initialize`
            (default top-1 accuracy + macro F1).
    """

    def __init__(
        self,
        model: Union[nn.Module, Fluid],
        dataset: Union[Dataset[Any], Fluid],
        experiment_name: str = "sonair_classification_eval",
        run_name: Optional[str] = None,
        label_map: Optional[LabelMap] = None,
        metrics: dict[str, Any] = {
            "accuracy_top1": LazyClass(MulticlassAccuracy, top_k=1),
            "f1_macro": LazyClass(MulticlassF1Score, average="macro"),
        },
    ) -> None:
        super().__init__(model=model, dataset=dataset, experiment_name=experiment_name, run_name=run_name)
        # Store config ONLY — no functional work. There is NO num_classes knob: for evaluation the
        # trained model's head is authoritative, so initialize() reads it off the LIVE model. `metrics`
        # holds single-metric LazyClass templates flowed with that count into the test/-prefixed
        # MetricCollection the base test_step reads. `label_map` encodes string targets (mirrors the trainer).
        self.num_classes: Optional[int] = None
        self.label_map = label_map
        self.metrics = metrics
        self.loader = LazyClass(DataLoader, collate_fn=classification_collate_fn)

    def run(self) -> None:
        """Resolve num_classes + flow the model and metrics, then run the base eval."""
        self.initialize()
        super().run()

    def initialize(self) -> None:
        """Flow the model, take ``num_classes`` from it, encode string targets, and build metrics.

        For evaluation the model is already trained — it loads its own checkpoint, so its class
        count is authoritative. We therefore **flow the model first** and read ``num_classes`` off
        the LIVE model (``getattr(model, "num_classes")``) — NOT from the eval dataset (which may be
        a subset and would mis-size the metrics against the model's head), and NOT from the model's
        deferred ``Fluid`` kwargs (those may be filled in later). There is no evaluator
        ``num_classes`` knob: the model is the single source of truth.

        When the eval dataset's targets are *strings*, they are encoded to class ids through the
        pinned :class:`~dataflux.LabelMap` — ``self.label_map`` if given, else reloaded from the
        ``class_names.json`` next to the model's checkpoint (the file marainer wrote at train time).
        We never refit on eval data: the training-time ordering is authoritative. The single-metric
        templates in ``self.metrics`` are then built into a ``test/``-prefixed ``MetricCollection``.
        Called explicitly from :meth:`run`. Idempotent.
        """
        # Flow the model first — it builds with its own num_classes (from config / the
        # checkpoint it loads). flow auto-solidifies a lazy TimmModel backbone.
        self.model = flow(self.model)
        # num_classes is derived from the trained MODEL (its head is authoritative for eval).
        nc = getattr(self.model, "num_classes", None)
        if not nc:
            raise RuntimeError(
                "LightningClassificationEvaluator: could not determine num_classes from the model — "
                "the model must expose a `num_classes` attribute (e.g. sonair.TimmModel does)."
            )
        self.num_classes = int(nc)

        # Normalize the eval source to a Dataset[Sample] up front (marainer helper) so target
        # detection / encoding / collate all see guaranteed Sample items.
        self.dataset = ensure_sample_dataset(flow(self.dataset))

        # Encode string targets through the pinned training map (mirrors the trainer).
        kind = detect_target_kind(self.dataset)
        if kind == "multilabel":
            raise NotImplementedError(
                "LightningClassificationEvaluator: multi-label string targets are not yet supported "
                "(needs multi-hot encoding + a BCE-style loss + collate changes). See TASKS.md."
            )
        if kind == "str":
            label_map = self.label_map or self._load_pinned_label_map()
            if label_map is None:
                raise RuntimeError(
                    "LightningClassificationEvaluator: eval dataset has string targets but no label_map "
                    "was given and no class_names.json was found next to the model checkpoint. Pass "
                    "`label_map`, or point the model at a checkpoint whose class_names.json exists."
                )
            self.label_map = label_map
            self.dataset = encode_flux(self.dataset, label_map)

        # Build the test/-prefixed collection via the shared helper (handles a dict of templates vs
        # a ready-made MetricCollection). Idempotent: once self.metrics is a test/-prefixed
        # MetricCollection, re-running initialize() is a no-op.
        if not (isinstance(self.metrics, MetricCollection) and getattr(self.metrics, "prefix", None) == "test/"):
            base = build_metric_collection(self.metrics, num_classes=self.num_classes)
            self.metrics = base.clone(prefix="test/")

    def _load_pinned_label_map(self) -> Optional[LabelMap]:
        """Reload the training-time :class:`~dataflux.LabelMap` from a checkpoint-adjacent class_names.json."""
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
        except Exception as exc:  # pragma: no cover - defensive: malformed/partial file
            logger.warning(f"Could not load LabelMap from {candidate}: {exc}")
            return None
