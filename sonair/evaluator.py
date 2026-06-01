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
      model: !class:sonair.timm_classifier.TimmModel
        model_name: efficientvit_b0
        num_classes: !ref:num_classes
        pretrained: false
        checkpoint_path: ./checkpoints/last.ckpt
      dataset: !class:dataflux.sources.HuggingFaceSource
        path: kitofrank/RFUAV
        split: validation
"""

from typing import Any

from confluid import LazyClass, configurable, flow
from confluid.fluid import Fluid
from marainer.evaluator import LightningEvaluator
from torch.utils.data import DataLoader

from sonair.classification import classification_collate_fn
from sonair.trainer import _build_classification_metrics


@configurable(task="classification", role="evaluator")
class LightningClassificationEvaluator(LightningEvaluator):
    """Specialised image-classification evaluator with baked-in defaults.

    Mirrors :class:`~marainer.evaluator.LightningEvaluator`'s minimal-ctor
    discipline: only ``model`` / ``dataset`` / ``num_classes`` (+ identity
    ``experiment_name`` / ``run_name``). The classification defaults — the
    ``test/``-prefixed metric set and the ``classification_collate_fn`` loader —
    are body attributes; ``predictions_sink`` / ``lightning`` /
    ``metrics_sidecar_path`` inherit the base defaults and are configured
    post-construction (YAML / broadcasting; a deferred slot uses ``!lazy:``).

    Args:
        model: Classification model that emits ``[N, num_classes]`` logits.
            The model is responsible for loading its own weights — pass
            ``checkpoint_path`` to the model class.
        dataset: Evaluation dataset or Flux pipeline.
        num_classes: Number of output classes; drives metric construction.
        experiment_name: MLflow experiment name.
        run_name: MLflow run name.
    """

    def __init__(
        self,
        model: Any,
        dataset: Any,
        num_classes: int,
        experiment_name: str = "sonair_classification_eval",
        run_name: str = "evaluate",
    ) -> None:
        super().__init__(model=model, dataset=dataset, experiment_name=experiment_name, run_name=run_name)
        self.num_classes = int(num_classes)
        self.metrics = _build_classification_metrics(self.num_classes).clone(prefix="test/")
        self.loader = LazyClass(DataLoader, collate_fn=classification_collate_fn)

    def run(self) -> None:
        """Flow the model with the (required) ``num_classes`` and realize it, then evaluate.

        sonair OWNS the num_classes injection here too (mirroring the trainer) instead of the
        generic ``marainer.dimensions`` path: a deferred model is flowed with the evaluator's
        ``num_classes`` and a lazy backbone (e.g. ``TimmClassifierForPrediction``) is realized.
        By the time the base ``run()`` executes, the model is LIVE so its generic derivation
        no-ops.
        """
        if isinstance(self.model, Fluid):
            self.model = flow(self.model, num_classes=self.num_classes)
        else:
            self.model = flow(self.model)
        realize = getattr(self.model, "realize", None)
        if callable(realize):
            realize()
        super().run()
