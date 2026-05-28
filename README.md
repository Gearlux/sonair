# Sonair

Image-classification models and adapters for the [Marainer](https://github.com/Gearlux/marainer) training framework.

Sonair provides a thin, Confluid-configurable wrapper around
[`timm`](https://github.com/huggingface/pytorch-image-models)
so any of timm's ~1000 pretrained backbones plug into the **unchanged** base
`marainer.trainer.LightningTrainer` — no subclassing, no overridden step functions, no
custom Lightning module. The Trainer's standard
`preds = self(batch.input); loss = self.loss_fn(preds, batch.target)` flow
fits classification natively.

The first-class model is **TimmClassifierModel**, a generic wrapper that
takes any `model_name` from timm's catalogue (efficientvit, resnet, vit,
swin, convnext, mobilenetv3, ...) and exposes its constructor knobs to YAML.

## Installation

```bash
pip install git+https://github.com/Gearlux/sonair.git@main
```

### Development

```bash
git clone https://github.com/Gearlux/sonair.git
cd sonair
pip install -e ".[dev]"
```

## Quick Start

### YAML wiring

```yaml
model: !class:sonair.timm_classifier.TimmClassifierModel
  model_name: efficientvit_b0
  num_classes: 37
  pretrained: true

loss_fn: !class:sonair.classification.ClassificationLoss
  label_smoothing: 0.1            # timm's modern-classifier default
  # class_weights: [...]          # optional — per-class re-weighting

val_metrics: !class:torchmetrics.classification.MulticlassAccuracy
  num_classes: 37

train_loader: !class:torch.utils.data.DataLoader
  batch_size: 32
  shuffle: true
  collate_fn: !ref:sonair.classification.classification_collate_fn

trainer: !class:marainer.trainer.LightningTrainer
  experiment_name: my_classification
  run_name: efficientvit_b0
  model: !ref:model
  loss_fn: !ref:loss_fn
  val_metrics: !ref:val_metrics
  ...
```

### Full end-to-end example

`waivefront-rfuav/config/train_efficientvit_b0.yaml` trains a 37-class
drone-type classifier on RFUAV spectrograms. It demonstrates the typical
pipeline:

1. `RegionsJsonSource` reads per-window annotations.
2. Waivefront ops render the IQ → spectrogram → image tensor.
3. `waivefront.classification_ops.MetadataLabelToClassificationTargetOp`
   reads `metadata["drone"]` and writes a scalar int64 class id.
4. `sonair.timm_classifier.TimmClassifierModel` consumes the image and
   emits per-image class logits.
5. `sonair.classification.ClassificationLoss` (CrossEntropyLoss) trains it.

Run:

```bash
marainer train waivefront-rfuav/config/train_efficientvit_b0.yaml
```

Sibling configs `evaluate_efficientvit_b0.yaml` and
`predict_efficientvit_b0.yaml` measure quality on the held-out test split
and dump per-window classification JSON over the full RFUAV dataset.

## Models

| Class | timm factory | Notes |
|-------|--------------|-------|
| `TimmClassifierModel` | `timm.create_model(model_name, ...)` | Generic — any timm backbone via `model_name` arg, default `efficientvit_b0` |

Recommended backbones for spectrogram classification:

| `model_name` | Native input | Speed | Notes |
|--------------|--------------|-------|-------|
| `efficientvit_b0` | 224×224 | fast | Modern ViT-CNN hybrid; good speed/accuracy balance |
| `resnet50` | 224×224 | medium | Strong baseline; widely benchmarked |
| `vit_base_patch16_224` | 224×224 | slow | Heavy but very accurate |
| `mobilenetv3_small_100` | 224×224 | very fast | Mobile/edge-friendly |
| `convnext_tiny` | 224×224 | medium | Modern CNN re-imagining |

See `timm.list_models()` for the full catalogue.

## Adapters

| Symbol | Purpose |
|--------|---------|
| `ClassificationLoss` | `nn.CrossEntropyLoss` wrapped for Confluid configuration; supports `class_weights`, `label_smoothing`, and `ignore_index`. |
| `classification_collate_fn` | DataLoader collate that stacks per-sample image tensors and scalar class-id targets, preserves per-sample metadata as a list (the default collate crashes on string-valued metadata). |
| `load_lightning_checkpoint` | Strip Lightning's `"model."` prefix when loading a `.ckpt` saved through `marainer.trainer.LightningTrainer`. |

Most torchmetrics classification metrics
(`MulticlassAccuracy`, `MulticlassF1Score`, `MulticlassAUROC`, ...) accept
logits directly and emit a scalar — they wire into `val_metrics` straight
from YAML, no adapter needed.

## Inherited from marainer

Sonair uses `marainer.trainer.LightningTrainer` unchanged, so training runs
automatically gain marainer's default checkpoint + YAML-logging behaviour:
a `ModelCheckpoint` (`save_last=True` + best by the first val metric) is
auto-injected when the YAML doesn't already provide one, and the full
config tree is logged as an artifact to every wired Lightning logger.
Opt into the per-logger `log_all_checkpoints` flag by referencing
`marainer.loggers.Marainer{TensorBoard,CSV,MLFlow}Logger` in YAML.
See marainer's README → *Default checkpointing* for details.

## Dependencies

| Package | Purpose |
|---------|---------|
| `marainer` | Base Trainer / Evaluator. |
| `confluid` | YAML-driven dependency injection (`@configurable`, `!class:`, `!ref:`). |
| `data-flux` | Sample / Flux pipeline plumbing. |
| `log-flow` | Structured logging. |
| `timm` | Underlying timm classification backbones. |
| `torch`, `torchmetrics`, `pytorch-lightning` | Standard PyTorch stack. |
