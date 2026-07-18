# Sonair

Image-classification models and adapters for the [Marainer](https://github.com/Gearlux/marainer) training framework.

Sonair provides a thin, Confluid-configurable wrapper around
[`timm`](https://github.com/huggingface/pytorch-image-models)
so any of timm's ~1000 pretrained backbones plug into the **unchanged** base
`marainer.trainer.LightningTrainer` — no subclassing, no overridden step functions, no
custom Lightning module. The Trainer's standard
`preds = self(batch.input); loss = self.loss_fn(preds, batch.target)` flow
fits classification natively.

The first-class model is **TimmModel**, a generic wrapper that
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
model: !class:sonair.TimmModel
  model_name: efficientvit_b0
  num_classes: 37
  pretrained: true

loss_fn: !class:torch.nn.CrossEntropyLoss   # the classification loss (no sonair wrapper)
  label_smoothing: 0.1            # timm's modern-classifier default

val_metrics: !class:torchmetrics.classification.MulticlassAccuracy
  num_classes: 37

train_loader: !lazy:torch.utils.data.DataLoader
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
3. For a raw-spectrogram pipeline, `sampleflux.ops.torch.ToTensorOp` tensorizes
   the image and `sampleflux.ops.target.MetadataToTargetOp` → `EncodeTargetOp`
   turn the metadata label into a scalar int64 class id (the declarative
   scikit-`LabelEncoder` analogue — there is no sonair/waivefront target op).
   The `sampleflux.sources.HuggingFaceSource` path skips step 3 entirely: HF's
   `ClassLabel` feature already yields an integer label.
4. `sonair.TimmModel` consumes the image and
   emits per-image class logits.
5. `torch.nn.CrossEntropyLoss` (registered as the classification loss) trains it.

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
| `TimmModel` | `timm.create_model(model_name, ...)` | Generic — any timm backbone via `model_name` arg, default `efficientvit_b0` |

Recommended backbones for spectrogram classification:

| `model_name` | Native input | Speed | Notes |
|--------------|--------------|-------|-------|
| `efficientvit_b0` | 224×224 | fast | Modern ViT-CNN hybrid; good speed/accuracy balance |
| `resnet50` | 224×224 | medium | Strong baseline; widely benchmarked |
| `vit_base_patch16_224` | 224×224 | slow | Heavy but very accurate |
| `mobilenetv3_small_100` | 224×224 | very fast | Mobile/edge-friendly |
| `convnext_tiny` | 224×224 | medium | Modern CNN re-imagining |

See `timm.list_models()` for the full catalogue.

`TimmModel` is tagged `@configurable(task="classification", role="model")` (the
trainer/evaluator carry `role="trainer"`/`"evaluator"`, and `torch.nn.CrossEntropyLoss`
is `register`-ed as `role="loss"`), so navigaitor's `list_classes(task=…, role=…)`
scan + the visual-editor form-spec offer them — and any future classification
model/loss — as task-scoped options. The dataset is the generic
`sampleflux.sources.HuggingFaceSource`.

## Class count: how `num_classes` is resolved

`LightningClassificationTrainer.initialize()` (run at the top of `run()`) resolves the count
in this order:

1. **A concrete (already-built) model is authoritative** — its classifier head is sized, so
   `num_classes` is read straight off it (`getattr(model, "num_classes")`). A dataset walk that
   disagreed would mis-size the metrics against the head.
2. An explicit `num_classes` on the trainer.
3. A `label_map` (see below).
4. A `!lazy:` model's pinned `num_classes` kwarg.
5. **Derived from the data** — `sampleflux.num_classes(train_set)` walks the (integer) targets and
   returns `max(class_id) + 1`.

So the canonical flat config wires the model `!lazy:` *without* `num_classes` and lets the
dataset drive it:

```yaml
model: !lazy:sonair.TimmModel        # !lazy: so the derived count is injected
  model_name: efficientvit_b0
train_set: !class:sampleflux.sources.HuggingFaceSource
  path: kitofrank/RFUAV
  split: train
```

## String labels, encoding, and balanced loss

When the dataset's `target` is a **class-name string** (a generic HF imagefolder, etc.) rather
than an integer id, the trainer auto-encodes it:

- **Encoding** — it fits a `sampleflux.LabelMap` from the train targets (scikit-learn
  `LabelEncoder`, shown with a progress bar), or uses a `label_map` you pass in. The wired
  `train`/`val`/`test` sets are wrapped so their targets become ids, and the map's class names are
  persisted to `class_names.json` next to the checkpoint — which `LightningClassificationEvaluator`
  reloads (via `sampleflux.LabelMap.load`) so eval/predict share the identical name↔id ordering.
- **Balanced loss** — with `balance_classes: true` (the default), the trainer derives
  inverse-frequency class weights from the targets it walked and injects them into the loss (when
  the loss takes a `weight` tensor, e.g. `CrossEntropyLoss`). The weights are exposed on
  `trainer.class_weights`. Set `balance_classes: false` to leave the loss unweighted.

```yaml
trainer: !class:sonair.LightningClassificationTrainer
  run_name: cats_vs_dogs
  balance_classes: true            # inverse-frequency class weights (default)
  # Optionally pin the encoding instead of fitting it:
  # label_map: !class:sampleflux.LabelMap
  #   mapping: { cat: 0, dog: 1 }
model: !lazy:sonair.TimmModel
  model_name: efficientvit_b0
train_set: !class:sampleflux.sources.HuggingFaceSource
  path: ./my_imagefolder      # string labels — auto-encoded
  split: train
```

Multi-label string targets (a *sequence* of names) are not yet supported and raise a clear error.

## Adapters

| Symbol | Purpose |
|--------|---------|
| `torch.nn.CrossEntropyLoss` | The default classification loss — registered (no sonair wrapper) as `task="classification", role="loss"`; set `label_smoothing` / `ignore_index` / `reduction` from YAML. Consumes int64 class-id targets `[N]`. |
| `SigmoidFocalLoss` | Thin `nn.Module` wrapper over `torchvision.ops.sigmoid_focal_loss` (a function, which confluid's `register` can't tag) — a pickable `task="classification", role="loss"` configurable with `alpha` / `gamma` / `reduction` knobs. Expects **float** targets the same shape as the logits (e.g. one-hot `[N, C]`), like the `BCEWithLogitsLoss` family. Defaults `reduction="mean"` (a scalar loss), not torchvision's `"none"`. |
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
Wire stock `pytorch_lightning.loggers.*` classes under `lightning.logger:`.
See marainer's README → *Default checkpointing* for details.

The classification trainer and evaluator also inherit marainer's post-run
`@output` I/O contract — `computed_metrics` (`Dict[str, float]`) and
`label_names` (`List[str]`) — so a run's scalar metrics and class names show up
in FluxStudio's Run-node results panel automatically (the classification trainer
adds its own wireable `trained_model` output on top). See marainer's README →
*Outputs (the `@output` I/O contract)*.

## Dependencies

| Package | Purpose |
|---------|---------|
| `marainer` | Base Trainer / Evaluator. |
| `confluid` | YAML-driven dependency injection (`@configurable`, `!class:`, `!ref:`). |
| `sampleflux` | Sample / Flux pipeline plumbing. |
| `loggair` | Structured logging. |
| `timm` | Underlying timm classification backbones. |
| `torch`, `torchmetrics`, `pytorch-lightning` | Standard PyTorch stack. |
