# Sonair

Image-classification models and a self-contained Lightning classifier, wired with
[Confluid](https://github.com/Gearlux/confluid) and configured from YAML.

Sonair provides a thin, Confluid-configurable wrapper around
[`timm`](https://github.com/huggingface/pytorch-image-models)
so any of timm's ~1000 pretrained backbones drop into **`sonair.LightningClassifier`** —
ONE standard `pytorch_lightning.LightningModule` with its own
`training_step` / `validation_step` / `test_step` / `predict_step` /
`configure_optimizers`, and a `run()` that dispatches on a `task` knob
(`fit` / `evaluate` / `test` / `predict`). Its standard
`preds = self(self._batch_input(batch)); loss = self.loss_fn(preds, self._batch_target(batch))`
flow — reading the primary input and target-`Label` fields off the batched typed
record — a plain dict of typed values — fits classification natively. Marainer is now used only for OPTIONAL experiment
tracking plus a couple of torch utilities.

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
  collate_fn: !ref:recordstream.collate_records

runnable: !class:sonair.LightningClassifier
  task: fit
  experiment_name: my_classification
  run_name: efficientvit_b0
  model: !ref:model
  loss_fn: !ref:loss_fn
  val_metrics: !ref:val_metrics
  ...
```

The config roots at a top-level `runnable:` key carrying a `task:` — the same
config serves training, evaluation, and prediction by switching the task (or by
picking the matching `sonair` verb, which wins over the YAML's `task`).

### Full end-to-end example

`examples/train_efficientvit_b0_mnist.yaml` trains an efficientvit classifier on
MNIST from a minimal flat config. A signal-domain pipeline (e.g. a drone-type
classifier over RFUAV spectrograms) follows the same shape and demonstrates the
typical stages:

1. `RegionsJsonSource` reads per-window annotations.
2. Waivefront ops render the IQ → spectrogram → image tensor.
3. For a raw-spectrogram pipeline, `recordstream.ops.torch.ToTensor` tensorizes
   the image and `recordstream.ops.target.MetadataToTargetOp` → `EncodeTargetOp`
   turn the metadata label into a scalar int64 class id (the declarative
   scikit-`LabelEncoder` analogue — there is no sonair/waivefront target op).
   The `recordstream.sources.HuggingFaceSource` path skips step 3 entirely: HF's
   `ClassLabel` feature already yields an integer label.
4. `sonair.TimmModel` consumes the image and
   emits per-image class logits.
5. `torch.nn.CrossEntropyLoss` (registered as the classification loss) trains it.

Run:

```bash
sonair train examples/train_efficientvit_b0_mnist.yaml
# or the generic runner:
recordstream run examples/train_efficientvit_b0_mnist.yaml
```

The `sonair evaluate <cfg>` and `sonair predict <cfg>` verbs bind the same
`LightningClassifier` with `task: evaluate` / `test` / `predict` to measure
quality on a held-out split or dump per-record classification JSON — the verb
wins over any `task:` set in the config, so one config can serve all three.

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
`LightningClassifier` carries `role="trainer"` — its per-task `run()` entry points
declare `trainer`/`evaluator`/`predictor` roles — and `torch.nn.CrossEntropyLoss`
is `register`-ed as `role="loss"`), so navigaitor's `list_classes(task=…, role=…)`
scan + the visual-editor form-spec offer them — and any future classification
model/loss — as task-scoped options. The dataset is the generic
`recordstream.sources.HuggingFaceSource`.

## Class count: how `num_classes` is resolved

`LightningClassifier.initialize()` (run at the top of `run()`) resolves the count
in this order:

1. **A concrete (already-built) model is authoritative** — its classifier head is sized, so
   `num_classes` is read straight off it (`getattr(model, "num_classes")`). A dataset walk that
   disagreed would mis-size the metrics against the head.
2. An explicit `num_classes` on the trainer.
3. A `label_map` (see below).
4. A `!lazy:` model's pinned `num_classes` kwarg.
5. **Derived from the data** — `recordstream.num_classes(train_set)` walks the (integer) targets and
   returns `max(class_id) + 1`.

So the canonical flat config wires the model `!lazy:` *without* `num_classes` and lets the
dataset drive it:

```yaml
model: !lazy:sonair.TimmModel        # !lazy: so the derived count is injected
  model_name: efficientvit_b0
train_set: !class:recordstream.sources.HuggingFaceSource
  path: kitofrank/RFUAV
  split: train
```

## String labels, encoding, and balanced loss

When the dataset's `target` is a **class-name string** (a generic HF imagefolder, etc.) rather
than an integer id, `initialize()` auto-encodes it:

- **Encoding** — it fits a `recordstream.LabelMap` from the train targets (scikit-learn
  `LabelEncoder`, shown with a progress bar), or uses a `label_map` you pass in. The wired
  `train`/`val`/`test` sets are wrapped so their targets become ids, and the map's class names are
  persisted to `class_names.json` next to the checkpoint — which a `task: evaluate` / `test` /
  `predict` run reloads (via `recordstream.LabelMap.load`) so eval/predict share the identical
  name↔id ordering.
- **Balanced loss** — with `balance_classes: true` (the default), `initialize()` derives
  inverse-frequency class weights from the targets it walked and injects them into the loss (when
  the loss takes a `weight` tensor, e.g. `CrossEntropyLoss`). The weights are exposed on
  `classifier.class_weights`. Set `balance_classes: false` to leave the loss unweighted.

```yaml
runnable: !class:sonair.LightningClassifier
  task: fit
  run_name: cats_vs_dogs
  balance_classes: true            # inverse-frequency class weights (default)
  # Optionally pin the encoding instead of fitting it:
  # label_map: !class:recordstream.LabelMap
  #   mapping: { cat: 0, dog: 1 }
model: !lazy:sonair.TimmModel
  model_name: efficientvit_b0
train_set: !class:recordstream.sources.HuggingFaceSource
  path: ./my_imagefolder      # string labels — auto-encoded
  split: train
```

Multi-label string targets (a *sequence* of names) are not yet supported and raise a clear error.

## Adapters

| Symbol | Purpose |
|--------|---------|
| `torch.nn.CrossEntropyLoss` | The default classification loss — registered (no sonair wrapper) as `task="classification", role="loss"`; set `label_smoothing` / `ignore_index` / `reduction` from YAML. Consumes int64 class-id targets `[N]`. |
| `SigmoidFocalLoss` | Thin `nn.Module` wrapper over `torchvision.ops.sigmoid_focal_loss` (a function, which confluid's `register` can't tag) — a pickable `task="classification", role="loss"` configurable with `alpha` / `gamma` / `reduction` knobs. Expects **float** targets the same shape as the logits (e.g. one-hot `[N, C]`), like the `BCEWithLogitsLoss` family. Defaults `reduction="mean"` (a scalar loss), not torchvision's `"none"`. |
| `recordstream.collate_records` | The generic DataLoader collate the classifier wires in (no bespoke sonair collate). Stacks the already-tensorized input under `input_key` (default `image`) to `[N, 3, H, W]` and leaves the other keys as per-record lists; the model boundary stacks the `Label` target under `target_key` (default `class`) into an `[N]` int64 tensor. |
| `load_lightning_checkpoint` | `marainer.torch` helper that strips Lightning's `"model."` prefix when loading a `.ckpt` saved by `sonair.LightningClassifier`. |

Most torchmetrics classification metrics
(`MulticlassAccuracy`, `MulticlassF1Score`, `MulticlassAUROC`, ...) accept
logits directly and emit a scalar — they wire into `val_metrics` straight
from YAML, no adapter needed.

## Optional tracking via marainer

`LightningClassifier` is a standard `pytorch_lightning.LightningModule` that builds
its own `pytorch_lightning.Trainer` from the `lightning` slot / ergonomic knobs. It
uses marainer ONLY for OPTIONAL experiment tracking: when `trackers` are wired it
attaches a `marainer.lightning.TrackingCallback` (git/config/log-file provenance),
and `log_system_metrics: true` activates framework-free system-metric logging.
Removing marainer would take only a few line changes. Wire stock
`pytorch_lightning.loggers.*` classes under the `logger:` knob (a list broadcasts).

`LightningClassifier` declares three read-only post-run `@output`s directly —
`trained_model` (the live trained `nn.Module`), `computed_metrics`
(`Dict[str, float]`), and `label_names` (`List[str]`) — so a run's model, scalar
metrics, and class names show up as wireable output sockets in StreamStudio's
Run-node results panel automatically.

## Dependencies

| Package | Purpose |
|---------|---------|
| `marainer` | Optional experiment tracking + torch utilities (`ensure_record_dataset` / `load_lightning_checkpoint`). |
| `confluid` | YAML-driven dependency injection (`@configurable`, `!class:`, `!ref:`). |
| `recordstream` | Record / Stream pipeline plumbing. |
| `loggair` | Structured logging. |
| `timm` | Underlying timm classification backbones. |
| `torch`, `torchmetrics`, `pytorch-lightning` | Standard PyTorch stack. |
