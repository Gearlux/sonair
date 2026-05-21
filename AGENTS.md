# Sonair Mandates

- **Adapter, Not Classifier:** Sonair is a thin Confluid-`@configurable` wrapper around `timm`. Any of timm's ~1000 pretrained backbones MUST plug into the **unchanged** `marainer.trainer.LightningTrainer` step flow — no subclassing the Trainer, no overridden step functions, no custom Lightning module.
- **Per-Sample Predictions for Sinks:** `TimmClassifierForPrediction` MUST emit per-sample dict outputs (`{prediction, logits, ...}`) so marainer's `Evaluator.predictions_sink` can stream them downstream. Don't return a batched tensor where a per-sample mapping is expected.
- **Generic Primitives Live in Waivefront:** Reusable target ops (`MetadataLabelToClassificationTargetOp`) and predictions sinks (`ClassificationPredictionsSink`) belong in `waivefront`, not `sonair`. Sonair stays focused on the model + loss + collate trio.
- **Loss Knobs Explicit:** `ClassificationLoss` MUST expose `label_smoothing` and `class_weights` via YAML — timm-era classifier defaults that users routinely tune.
- **Collate Discipline:** `classification_collate_fn` MUST preserve per-sample metadata alongside the batched input/target tensors so sinks can correlate predictions back to source samples.
- **Backbone Knobs Pass-Through:** `TimmClassifierModel.__init__` MUST forward `model_name`, `num_classes`, `pretrained`, and any timm constructor knobs verbatim to `timm.create_model`. Don't shadow names; don't silently rewrite defaults.
