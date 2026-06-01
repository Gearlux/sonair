"""LightningClassificationEvaluator: specialised image-classification evaluator.

Subclass of :class:`~marainer.evaluator.LightningEvaluator` that bakes in
classification-specific defaults so a minimal YAML config only needs to specify
the dataset, model, and number of classes.

Baked-in defaults (all overridable via YAML ``@configurable`` attributes):

* ``loader`` ``collate_fn`` — :func:`~sonair.classification.classification_collate_fn`
* ``metrics`` — :class:`~torchmetrics.classification.MulticlassAccuracy` (top-1)
  + :class:`~torchmetrics.classification.MulticlassF1Score` (macro);
  top-5 accuracy added when ``num_classes >= 5``.

Minimal YAML example::

    evaluator: !class:sonair.evaluator.LightningClassificationEvaluator
      num_classes: 37
      model: !class:sonair.timm_classifier.TimmClassifierModel
        model_name: efficientvit_b0
        num_classes: !ref:num_classes
        pretrained: false
        checkpoint_path: ./checkpoints/last.ckpt
      dataset: !class:sonair.hf_dataset.HuggingFaceClassificationDataset
        repo_id: kitofrank/RFUAV
        split: validation
"""

from typing import Any, Optional

import pytorch_lightning as L
from confluid import Class, Lazy, configurable
from marainer.evaluator import LightningEvaluator
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection

from sonair.classification import classification_collate_fn
from sonair.trainer import _build_classification_metrics


@configurable(task="classification", role="evaluator")
class LightningClassificationEvaluator(LightningEvaluator):
    """Specialised image-classification evaluator with baked-in defaults.

    Args:
        model: Classification model that emits ``[N, num_classes]`` logits.
            The model is responsible for loading its own weights — pass
            ``checkpoint_path`` to the model class.
        dataset: Evaluation dataset or Flux pipeline.
        num_classes: Number of output classes; drives metric construction.
        loader: Deferred DataLoader (default includes ``classification_collate_fn``).
        predictions_sink: Optional sink with ``.write(prediction, metadata)`` and
            ``.close()``. Use :class:`~marainer.sinks.ClassificationPredictionsSink`
            for per-sample JSON output.
        lightning: PyTorch Lightning Trainer config.
        experiment_name: MLflow experiment name.
        run_name: MLflow run name.
        metrics_sidecar_path: Optional path; when set, writes a logger-agnostic
            JSON dump of the final metrics after evaluation completes.
    """

    def __init__(
        self,
        model: Any,
        dataset: Any,
        num_classes: int,
        loader: Lazy[Any] = Class(DataLoader, collate_fn=classification_collate_fn),
        predictions_sink: Any = None,
        lightning: Any = Class(L.Trainer),
        experiment_name: str = "sonair_classification_eval",
        run_name: str = "evaluate",
        metrics_sidecar_path: Optional[str] = None,
    ) -> None:
        metrics: MetricCollection = _build_classification_metrics(num_classes).clone(prefix="test/")
        super().__init__(
            model=model,
            dataset=dataset,
            loader=loader,
            metrics=metrics,
            predictions_sink=predictions_sink,
            lightning=lightning,
            experiment_name=experiment_name,
            run_name=run_name,
            metrics_sidecar_path=metrics_sidecar_path,
        )
