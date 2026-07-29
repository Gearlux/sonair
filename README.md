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
`preds = self(batch_tensor(batch, self.input_key)); loss = self.loss_fn(preds, _batch_target(batch, self.target_key))`
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
sonair lightning train examples/train_efficientvit_b0_mnist.yaml
# or the generic runner (no framework verb):
recordstream run examples/train_efficientvit_b0_mnist.yaml
```

That one config carries **every** backend: engine-agnostic keys (dataset, batch
size, epochs) at the top level, and one `!scope:framework=<name>` block per
alternative overriding only `runnable` / `model` / `logger`. The backend verb
activates the matching scope, so you never name the framework twice.

The `sonair lightning evaluate <cfg>` and `sonair lightning predict <cfg>` verbs
bind the same `LightningClassifier` with `task: test` / `predict` to measure
quality on a held-out split or dump per-record classification JSON — the verb
wins over any `task:` set in the config, so one config can serve all three.

## Training backends

Two engines, one config. `sonair <backend> <verb>` picks the engine; everything above the
config's framework divider — dataset, splits, batch size, epochs, run identity — is shared
verbatim.

```bash
sonair lightning train examples/train_efficientvit_b0_mnist.yaml   # PyTorch Lightning
sonair keras     train examples/train_efficientvit_b0_mnist.yaml   # Keras 3
sonair tensorflow train cfg.yaml                                   # template — names the work
```

| Backend | Status | Model wrapper | Notes |
|---|---|---|---|
| `lightning` | implemented | `TimmModel` (timm) | the reference backend |
| `keras` | implemented | `KerasModel` (`keras.applications`) | Keras 3 — see below |
| `tensorflow` / `jax` / `huggingface` / `mlx` | templates | — | the verbs exist and name the work |

### What a backend swap actually costs

The `keras` block in the example config overrides four keys and nothing else: `runnable`,
`model`, `logger`, and `preprocess`. Only that last one is a *data* difference, and it is a
real one — **Keras is channels-last, torch is channels-first** — so the Keras chain stops
after `Normalize` (leaving `HWC`) where the torch chain adds `ToTensor` to transpose to `CHW`.

Everything else is shared because it belongs to neither engine: the record stream, the
`LabelMap` that encodes string labels, the class-name vocabulary, the predictions sink, the
metrics sidecar. The Keras backend's own code is a ~20-line `PyDataset` adapter plus
`compile()`/`fit()` — that ratio is the demonstration.

### Choosing the Keras compute backend

Keras 3 runs on PyTorch, TensorFlow or JAX, chosen by `KERAS_BACKEND` **at import time**.
sonair defaults it to `torch` (the one it always installs) when you have not chosen, so
`sonair keras train` works out of the box:

```bash
sonair keras train cfg.yaml                          # torch (default)
KERAS_BACKEND=tensorflow sonair keras train cfg.yaml # the same config on TensorFlow
```

Install with `pip install 'sonair[keras]'` — Keras alone, four small packages. Installing
TensorFlow is a heavier decision (it moves protobuf a major version across mlflow, tensorboard
and wandb); see `TASKS.md`.


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

- **Encoding** — it fits a `recordstream.LabelMap` from the train targets (deterministic
  sorted-unique ordering, shown with a progress bar), or uses a `label_map` you pass in. The wired
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

## Multi-label classification

A record whose target is a `recordstream.MultiLabel` (several classes at once) needs no extra
configuration — `initialize()` detects it from the data and switches three things at once:

| | single-label (`Label`) | multi-label (`MultiLabel`) |
|---|---|---|
| target tensor | `[N]` int64 class ids | `[N, C]` **float** multi-hot |
| default `loss_fn` | `torch.nn.CrossEntropyLoss` | `torch.nn.BCEWithLogitsLoss` |
| default metrics | `Multiclass*` sized by `num_classes` | `Multilabel*` sized by `num_labels` |

Class-name strings are encoded exactly as in the single-label case (the same `LabelMap`, fitted
over every member of every record's label set), and `balance_classes` counts each of a record's
labels, so a record with two labels contributes to both classes' frequencies.

```yaml
runnable: !class:sonair.LightningClassifier
  task: fit
  # nothing multi-label-specific: MultiLabel targets in the data are enough.
  # Override either default explicitly if you want something else:
  # loss_fn: !class:sonair.classification.SigmoidFocalLoss { alpha: 0.25 }
  # metrics:
  #   exact_match: !lazy:sonair.classification.MultilabelExactMatch
```

Wiring a `loss_fn` or `metrics` yourself always wins — the defaults only fill an **unset** slot
(both are `None` until `initialize()` runs, which is how "not chosen" stays distinguishable from
"chosen"; see [docs/architecture.md](docs/architecture.md) §3).

The prediction sink still writes top-k entries per record; a threshold-based multi-label sink is
open work (see `TASKS.md`).

## What `predict` returns

`task: predict` emits a `recordstream.ClassificationOutput` per batch — not raw logits:

| Key | Shape | Meaning |
|---|---|---|
| `logits` | `[N, C]` | raw pre-softmax scores |
| `probs` | `[N, C]` | softmax over the class dim |
| `class_idx` | `[N]` int64 | argmax |

A predictions sink reads `probs` / `class_idx` **by name** — it cannot softmax on the caller's
behalf, since only the model knows whether its head already did. Wire one under
`predictions_sink` and the classifier slices the batch row-wise, writing one entry per record
paired with that record's metadata:

```yaml
runnable: !class:sonair.LightningClassifier
  task: predict
  predictions_sink: !class:recordstream.predictions.ClassificationPredictionsSink
    top_k: 3
    ops:
      - !class:recordstream.ops.sink.RecordSinkOp
        sink: ...
```

## Adapters

| Symbol | Purpose |
|--------|---------|
| `torch.nn.CrossEntropyLoss` | The default classification loss — registered (no sonair wrapper) as `task="classification", role="loss"`; set `label_smoothing` / `ignore_index` / `reduction` from YAML. Consumes int64 class-id targets `[N]`. |
| `SigmoidFocalLoss` | Thin `nn.Module` wrapper over `torchvision.ops.sigmoid_focal_loss` (a function, which confluid's `register` can't tag) — a pickable `task="classification", role="loss"` configurable with `alpha` / `gamma` / `reduction` knobs. Expects **float** targets the same shape as the logits (e.g. one-hot `[N, C]`), like the `BCEWithLogitsLoss` family. Defaults `reduction="mean"` (a scalar loss), not torchvision's `"none"`. |
| `torch.nn.BCEWithLogitsLoss` | The default **multi-label** loss (also registered, `role="loss"`). Consumes float multi-hot targets `[N, C]` shaped like the logits. |
| `recordstream.collate_records` | The generic DataLoader collate the classifier wires in (no bespoke sonair collate). Stacks the already-tensorized input under `input_key` (default `image`) to `[N, 3, H, W]` and leaves the other keys as per-record lists; the model boundary reads it back with `recordstream.batch_tensor` / `batch_metadata` and stacks the `Label` target under `target_key` (default `class`) into an `[N]` int64 tensor — or builds an `[N, C]` float multi-hot from a `MultiLabel`. |
| `load_lightning_checkpoint` | `marainer.torch` helper that strips Lightning's `"model."` prefix when loading a `.ckpt` saved by `sonair.LightningClassifier`. |

Most torchmetrics classification metrics accept logits directly and emit a
scalar — they wire into `val_metrics` straight from YAML, no adapter needed.
Both families are registered: `Multiclass*` (`MulticlassAccuracy`,
`MulticlassF1Score`, `MulticlassAUROC`, ...), sized by `num_classes`, and
`Multilabel*` (`MultilabelAccuracy`, `MultilabelF1Score`,
`MultilabelExactMatch`, ...), sized by `num_labels`. The classifier picks the
family that matches the data; name one explicitly to override.

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
(`Dict[str, float]`), and `class_names` (`List[str]`) — so a run's model, scalar
metrics, and class names show up as wireable output sockets in StreamStudio's
Run-node results panel automatically.

## Dependencies

| Package | Purpose |
|---------|---------|
| `marainer` | Optional experiment tracking (`load_lightning_checkpoint` / `load_lightning_checkpoint`). |
| `confluid` | YAML-driven dependency injection (`@configurable`, `!class:`, `!ref:`). |
| `recordstream` | Record / Stream pipeline plumbing. |
| `loggair` | Structured logging. |
| `timm` | Underlying timm classification backbones. |
| `torch`, `torchmetrics`, `pytorch-lightning` | Standard PyTorch stack. |
