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

**Pick your backend.** The core is framework-free — a training framework arrives with the
backend you select, so a Keras install does not drag in Lightning, nor the reverse:

```bash
pip install 'git+https://github.com/Gearlux/sonair.git@main#egg=sonair[lightning]'       # PyTorch Lightning
pip install 'git+https://github.com/Gearlux/sonair.git@main#egg=sonair[keras]'           # Keras 3
pip install 'git+https://github.com/Gearlux/sonair.git@main#egg=sonair[tensorflow]'      # native TensorFlow
pip install 'git+https://github.com/Gearlux/sonair.git@main#egg=sonair[fastai]'          # fastai Learner
pip install 'git+https://github.com/Gearlux/sonair.git@main#egg=sonair[huggingface]'     # transformers Trainer
pip install 'git+https://github.com/Gearlux/sonair.git@main#egg=sonair[jax]'             # jitted optax loop
pip install 'git+https://github.com/Gearlux/sonair.git@main#egg=sonair[mlx]'             # Apple-silicon MLX
pip install 'git+https://github.com/Gearlux/sonair.git@main#egg=sonair[lightning,keras]' # any combination
```

`[lightning]` is the reference backend and what the examples assume. What each selects:

| install | adds |
|---|---|
| `sonair` | nothing — CLI and config surface only |
| `sonair[lightning]` | `torch`, `pytorch-lightning`, `timm`, `torchmetrics` |
| `sonair[torch]` | the same MINUS `pytorch-lightning` — a hand-written loop and nothing under it |
| `sonair[keras]` | `keras` (Keras 3 — picks torch / tensorflow / jax at run time) |
| `sonair[tensorflow]` | `keras` + the TensorFlow engine |
| `sonair[fastai]` | `fastai`, `torch`, `timm` (no Lightning — the two trainers are alternatives) |
| `sonair[huggingface]` | `transformers`, `accelerate` (the Hub zoo + `Trainer`) |
| `sonair[jax]` | `jax`, `flax`, `optax` (no zoo — the shared conv spec, written in Flax) |
| `sonair[mlx]` | `mlx` (Apple silicon only — empty elsewhere, so the install still succeeds) |

Two of those extras carry an upper bound, and both are upstream breaks rather than
preferences: `sonair[fastai]` pins `fastcore<2` (fastai's own optimizer still calls
`L.starmap`, which fastcore 2.x removed) and `fastprogress<1.1` (1.1 added a
`python-fasthtml` dependency — a progress bar pulling in a web framework — and requires the
fastcore major this cannot use). Each is commented in `pyproject.toml` with what to observe
before dropping it.

Touching a backend you did not install says so, with the command to fix it:

```console
>>> import sonair; sonair.LightningClassifier
ImportError: sonair.LightningClassifier needs the 'lightning' backend, which is not installed.
  pip install 'sonair[lightning]'
```

### Development

```bash
git clone https://github.com/Gearlux/sonair.git
cd sonair
uv pip install -e '.[lightning,dev]'      # or '.[keras,dev]', or '.[lightning,keras,dev]'
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

`train_loader` / `val_loader` / `test_loader` are deferred slots: `batch_size` and
`num_workers` are the ergonomic knobs, but wiring the loader itself replaces it
wholesale — that is how you reach `pin_memory`, `drop_last`, or a sampler. The
**fastai** backend declares the same three slots under the same names (it consumes
plain torch loaders), so this block drives either backend unchanged. Its only
difference is the default `collate_fn`, which is `_to_tuple` rather than
`collate_records`, because fastai unpacks a batch positionally.

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
   The `recordstream.sources.huggingface.HuggingFaceSource` path skips step 3 entirely: HF's
   `ClassLabel` feature already yields an integer label.
4. `sonair.TimmModel` consumes the image and
   emits per-image class logits.
5. `torch.nn.CrossEntropyLoss` (registered as the classification loss) trains it.

Run:

```bash
sonair train examples/train_efficientvit_b0_mnist.yaml
# or the generic runner (no verb):
recordstream run examples/train_efficientvit_b0_mnist.yaml
```

That one config carries **every** backend: engine-agnostic keys (dataset, batch
size, epochs) at the top level, and one `!scope:framework=<name>` block per
alternative overriding only `runnable` / `model`. `--framework <name>`
activates the matching block — the same flag on either runner, because it is a
property of the config rather than of the CLI.

The `sonair evaluate <cfg>` and `sonair predict <cfg>` verbs bind the same
`LightningClassifier` with `task: test` / `predict` to measure quality on a
held-out split or dump per-record classification JSON — the verb wins over any
`task:` set in the config, so one config can serve all three.

## Training backends

Eight engines, one config. `--framework <name>` picks the engine; everything above the
config's framework divider — dataset, splits, batch size, epochs, run identity — is shared
verbatim.

```bash
sonair train examples/train_efficientvit_b0_mnist.yaml                          # PyTorch Lightning (default)
sonair train examples/train_efficientvit_b0_mnist.yaml --framework torch        # a hand-written loop, no framework
sonair train examples/train_efficientvit_b0_mnist.yaml --framework keras        # Keras 3
sonair train examples/train_efficientvit_b0_mnist.yaml --framework tensorflow   # tf.data + GradientTape
sonair train examples/train_efficientvit_b0_mnist.yaml --framework fastai       # Learner + one-cycle
sonair train examples/train_efficientvit_b0_mnist.yaml --framework huggingface  # transformers Trainer
sonair train examples/train_efficientvit_b0_mnist.yaml --framework jax          # jitted optax loop
sonair train examples/train_efficientvit_b0_mnist.yaml --framework mlx          # Apple-silicon unified memory

sonair frameworks                                                               # the list, with a line each
```

The name must match a block the config declares — `--framework kears` is an error
listing the backends that file *does* declare, not a silent fall-through to the
default. That check is the reason the backend is a flag: it used to be a
subcommand (`sonair keras train cfg.yaml`), which named an engine nothing ever
compared against the config.

| Backend | Model wrapper | Owns the loop with | Adds |
|---|---|---|---|
| `lightning` | `TimmModel` (timm), or `TorchConvNet` (the shared spec) | `pytorch_lightning.Trainer` | the reference implementation |
| `torch` | `TimmModel` / `TorchConvNet` — the same ones | a hand-written loop | the control experiment: no framework at all |
| `keras` | `KerasModel` (`keras.applications`) | `keras.Model.fit` | one API over torch / TF / JAX |
| `tensorflow` | `KerasModel` — the same one | a traced `tf.GradientTape` loop | `tf.data`, XLA, SavedModel export |
| `fastai` | `TimmModel` / `TorchConvNet` — the same ones | `Learner.fit_one_cycle` | fastai's schedule and callbacks |
| `huggingface` | `HuggingFaceModel` (the Hub) | `transformers.Trainer` | fine-tuning a pretrained checkpoint |
| `jax` | `FlaxConvNet` (the shared spec) | a jitted `optax` loop | functional transforms; no zoo to wrap |
| `mlx` | `MlxConvNet` (the shared spec) | `mlx.optimizers` + `nn.value_and_grad` | unified memory; Apple silicon only |

On the shipped MNIST config (3 epochs), `lightning` reaches **99.52%** top-1 and `keras`
**99.18%** best val from the same file, changing only the flag. Different backbones, so this
is not a benchmark — it is evidence that one config drives several engines to a comparable
result.

### Running them all — the sweep

`examples/sweep/` trains every backend this machine has, twice each (once on a model-zoo
backbone and once on the shared conv spec), and reports the lot from one tracker:

```bash
examples/sweep/run_sweep.sh                     # every installed backend, into MLflow
examples/sweep/run_sweep.sh --tracker csv       # ...into the always-available CSV backend
examples/sweep/run_sweep.sh --only jax,mlx      # just those two
examples/sweep/run_sweep.sh --dry-run           # print the matrix and the commands, run nothing
```

Every run is the same `examples/sweep/all_backends.yaml` with two flags changed. That file is
an **overlay**: it `include:`s the example above and adds only what a sweep needs — the five
backend blocks the example keeps commented (they must load on a checkout that installs three
frameworks, and a sweep is opt-in), a convnet per engine, and a `--tracker` dimension.

The matrix, and why two cells are empty:

| Backend | zoo model | `--model convnet` |
|---|---|---|
| lightning · torch · fastai | `TimmModel` efficientvit_b0 | `TorchConvNet` |
| keras | `KerasModel` MobileNetV3Small | `KerasConvNet` |
| tensorflow | `KerasModel` MobileNetV3Small | `TensorFlowConvNet` |
| huggingface | `HuggingFaceModel` resnet-18 | — a `transformers.Trainer` drives models that compute their own loss |
| jax | — no Flax zoo exists | `FlaxConvNet` |
| mlx | — no MLX image zoo exists | `MlxConvNet` |

Those two gaps are the reason [`sonair/convnet/spec.py`](sonair/convnet/spec.py) exists: one
architecture, written natively per framework, so the comparison is between *frameworks* rather
than between five different networks.

The script builds its matrix by **probing** which frameworks import, not by reading
`sonair frameworks` (which advertises all eight regardless), so a machine with three extras
runs three backends and says which it skipped. Each run is its own process — not a style
choice: Keras 3 binds its compute engine at import, so `keras` and `tensorflow` cannot share an
interpreter. A backend that fails does not abort the others; its log is named in the summary.

The experiment is named `<Day>_<Mon>_<D>_<YYYY>_<N>`, N counted from the tracker itself, so a
wiped tracker restarts at 1. At the end the script prints one row per run — every metric any
backend recorded, since they genuinely disagree on naming (`val/accuracy_top1` under torch,
`val_accuracy` under Keras):

```
| run               | status   |   sec |   val/accuracy_top1 |   val_accuracy |   val/loss |
|-------------------|----------|-------|---------------------|----------------|------------|
| lightning_zoo     | FINISHED |    74 |              0.9727 |                |     0.1744 |
| lightning_convnet | FINISHED |    73 |              0.2857 |                |     2.1602 |
```

MLflow and CSV can be read back from here; the others are write surfaces from this machine and
the report says where to open them instead.

### What a backend swap actually costs

The `keras` block in the example config overrides four keys and nothing else: `runnable`,
`model` and `preprocess`. Only that last one is a *data* difference, and it is a
real one — **Keras is channels-last, torch is channels-first** — so the Keras chain stops
after `Normalize` (leaving `HWC`) where the torch chain adds `ToTensor` to transpose to `CHW`.

Everything else is shared because it belongs to neither engine: the record stream, the
`LabelMap` that encodes string labels, the class-name vocabulary, the predictions sink, the
metrics sidecar — and the batching, since `recordstream.keras.RecordSequence` is the
`keras.utils.PyDataset` a `DataLoader` would have been. The Keras backend's own code is an
eight-line batch-shaping function plus `compile()`/`fit()` — that ratio is the demonstration.

The other two blocks make the point twice more, from opposite directions. The `fastai` block
is the smallest of the four: it overrides `runnable`, and *nothing else* — no
`model`, no `preprocess`, because fastai runs on torch and consumes the reference backend's
`TimmModel` and its `[N, 3, H, W]` batches unchanged. The `tensorflow` block overrides no
`model` either, for the mirror reason: in TensorFlow ≥ 2.16 `tf.keras` **is** Keras 3, so
`KerasModel`, `keras.losses` and `keras.metrics` are the same objects the `keras` backend
uses.

### The model slot takes your own model

Each wrapper subclasses its engine's model base — `TimmModel` is an `nn.Module`, `KerasModel`
is a `keras.Model` — so the slot is not restricted to them. Anything of the right kind is
wired the same way, and a classifier drives it identically:

```yaml
# instead of `!lazy:sonair.KerasModel`, wire a keras.Model subclass of your own.
# Note the parens: the instance form builds it now, where `!lazy:` defers.
model: !class:my_project.MyClassifier()
```

The wrappers are worth using because they are LAZY: the network is built when the classifier
flows them, with the dataset-derived `num_classes` injected, so the class count never has to
be written in the config. A model that is already live keeps the width its own constructor was
given — the injection is skipped rather than applied — so size it yourself when you wire one.

#### Swapping the model from the command line

The model slot takes a scope block like any other key, so an alternative can sit beside the
default and be selected by a flag. The example config ships exactly that — the zoo-backed
`TimmModel` by default, the from-scratch `TorchConvNet` behind `model=convnet`:

```yaml
runnable: !class:sonair.LightningClassifier

model: !lazy:sonair.TimmModel
  model_name: efficientvit_b0
  pretrained: true

convnet: !scope:model=convnet
  model: !lazy:sonair.convnet.torch_convnet.TorchConvNet
```

```bash
sonair train examples/train_efficientvit_b0_mnist.yaml                  # 2.1 M params, pretrained
sonair train examples/train_efficientvit_b0_mnist.yaml --model convnet  # 95 K params, from scratch
sonair train examples/train_efficientvit_b0_mnist.yaml --framework torch --model convnet
```

The block must come *after* the slot it overrides — an active block splices in at its own
position and the later write wins. `--model convnet` is the implicit per-dimension flag for
the `model=` scope; `--scope model=convnet` is the explicit spelling of the same thing.

Both `model:` and its alternative sit at the **top level**, as siblings of `runnable:` rather
than nested inside it. That is what lets all three torch-backed backends (`lightning`,
`torch`, `fastai`) read the same line: a `!scope:framework=…` block that swaps `runnable:`
replaces that key and everything under it, so a nested `model:` would vanish for exactly the
backends whose whole point is that they inherit it.

### `tensorflow` — when to prefer it over `keras`

`KERAS_BACKEND=tensorflow sonair train cfg.yaml --framework keras` already runs the identical config on
TensorFlow, so the separate backend earns its place only by doing what Keras 3 does not
expose. It contains exactly that and nothing else:

- **`tf.data`** — `tf.data.Dataset.from_generator` over the record stream, with
  `.prefetch(tf.data.AUTOTUNE)` overlapping the record walk with the device step.
- **A traced loop** — `tf.GradientTape` under `@tf.function`, which is what makes
  `jit_compile: true` (XLA) a knob at all.
- **SavedModel** — `export_dir:` writes a `tf.saved_model` directory (serving signatures,
  frozen graph), the TensorFlow artifact with no Keras equivalent.

```yaml
runnable: !class:sonair.TensorFlowClassifier
  export_dir: ./exported/mnist      # writes saved_model.pb + variables/ after fit
  jit_compile: false                # XLA: faster on fixed shapes, real first-batch cost
  prefetch: true                    # tf.data.AUTOTUNE
```

Everything else — the label handling, the batch shape, the losses and metrics, the
prediction contract — is *inherited* from `KerasClassifier`, not re-implemented.

> **One engine per process.** Keras 3 binds its compute backend at first import, and
> `recordstream.keras` defaults it to the first engine actually installed — `torch`, on a
> machine that also has `sonair[lightning]`. `--framework tensorflow` sets
> `KERAS_BACKEND=tensorflow` before anything imports keras; if something else got there
> first, `initialize()` refuses to start and names the fix rather than failing inside the
> tape.

### `fastai` — the loop, and only the loop

```yaml
runnable: !class:sonair.FastaiClassifier
  learning_rate: 0.001
  one_cycle: true                   # fit_one_cycle (the default) vs a flat fit
  opt_func: Adam                    # a name in fastai.optimizer
  metrics: [accuracy, top_k_accuracy]
```

`metrics` are **names, not `!class:` tags**, and that is a property of fastai rather than a
shortcut taken here: `accuracy`, `error_rate` and `top_k_accuracy` are plain functions, and
`F1Score()` / `RocAuc()` are factories that *return* one — none of them is a class, so there
is nothing for a tag to resolve or for confluid's registry to tag. Names are resolved against
`fastai.metrics`, and an unknown one says where names come from. `opt_func` works the same
way against `fastai.optimizer`. Losses *are* classes and are registered normally (see below).

Two behaviours worth knowing: fastai's `Learner` **always** validates, so a config with no
`val_set` reuses the training data and logs a warning saying the reported metrics measure fit
rather than generalization; and `predict` runs a plain `torch.no_grad()` forward rather than
`Learner.get_preds`, because the sink contract needs each row paired with the metadata that
travelled beside the image.

### Choosing the Keras compute backend

Keras 3 runs on PyTorch, TensorFlow or JAX, chosen by `KERAS_BACKEND` **at import time**.
When you have not chosen, the first engine actually installed is used (`torch` for
`sonair[keras]`), so `--framework keras` works out of the box:

```bash
sonair train cfg.yaml --framework keras                          # torch (default)
KERAS_BACKEND=tensorflow sonair train cfg.yaml --framework keras # the same config on TensorFlow
```

Install with `pip install 'sonair[keras]'` — Keras alone, four small packages. `[tensorflow]`
adds the engine on top; it is a bigger download, but **not** a dependency conflict: a clean
resolve of TensorFlow alongside the tracking stack (wandb, mlflow, tensorboard,
opentelemetry-proto, grpcio-status) lands on a single protobuf everything accepts — 6.33.6,
measured 2026-07-30.

If you install it into a long-lived environment and `uv pip check` then complains about
protobuf, that is *incremental-install drift*, not TensorFlow: uv moves the minimum it must, so
protobuf jumps while older `wandb` / `grpcio-status` / `opentelemetry-proto` stay pinned behind
it. Every one of those has since relaxed to `protobuf<8`. Let them upgrade too, or recreate the
environment.


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
`recordstream.sources.huggingface.HuggingFaceSource`.

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
train_set: !class:recordstream.sources.huggingface.HuggingFaceSource
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
train_set: !class:recordstream.sources.huggingface.HuggingFaceSource
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
  # loss_fn: !class:sonair.torch_classification.SigmoidFocalLoss { alpha: 0.25 }
  # metrics:
  #   exact_match: !lazy:sonair.torch_classification.MultilabelExactMatch
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

## Losses and metrics

Sonair defines no loss and no metric. It **registers** upstream ones so they are pickable in
discovery surfaces and wirable from YAML — and it registers them **per engine**, because a
loss is engine-bound: a `torch.nn` loss cannot be handed to `KerasClassifier`, and a
`keras.losses` one cannot be handed to `LightningClassifier`.

| Engine | Module | Registered | Tag |
|---|---|---|---|
| PyTorch | [`sonair/torch_classification.py`](sonair/torch_classification.py) | `torch.nn` losses + `torchmetrics.classification` `Multiclass*` / `Multilabel*` | `framework="torch"` |
| Keras 3 | [`sonair/keras_classification.py`](sonair/keras_classification.py) | `keras.losses` + `keras.metrics` | `framework="keras"` |
| fastai | [`sonair/fastai_classification.py`](sonair/fastai_classification.py) | the `fastai.losses` classes (`CrossEntropyLossFlat`, `FocalLossFlat`, `LabelSmoothingCrossEntropy`, …) | `framework="torch"` |

Each module is its own `confluid.configurables` entry point, so an install without a given
backend simply has no registrations for it. A picker asked for "a classification loss" for a
Keras trainer is offered only the Keras set (confluid 0.3.0's `framework` axis) — see
[docs/architecture.md](docs/architecture.md) §2.

Two rows in that table are less symmetric than they look, and both are the `framework` axis
working as intended — it names the API a class can be **wired to**, not the package that
published it. The fastai losses are tagged `torch` because that is what they consume: a
`LightningClassifier` takes `CrossEntropyLossFlat` as happily as a `FastaiClassifier` does. And
there is deliberately **no** `tensorflow_classification.py`: in TensorFlow ≥ 2.16 `tf.keras`
*is* Keras 3, so `TensorFlowClassifier` uses the Keras set — a second module would register the
same classes under a second tag and make every bare `!class:SparseCategoricalCrossentropy`
ambiguous for no gain.

Every backend that owns a loss slot leaves `loss` / `metrics` unset by default and resolves
them from the data at `initialize()` (`Sparse`/`CrossEntropy` for int64 ids, the binary/`BCE` family for a
`MultiLabel` target). Everything below is for when you want to say it yourself.

### Per-split metrics

`metrics:` is what **every** split measures. `train_metrics:` / `val_metrics:` / `test_metrics:`
are what **that** split measures on top — merged over the shared set by name, so adding one
metric to one split does not mean restating the others beside it:

```yaml
metrics:                       # every split
  accuracy_top1: !lazy:torchmetrics.classification.MulticlassAccuracy { top_k: 1 }
  f1_macro: !lazy:torchmetrics.classification.MulticlassF1Score { average: macro }

val_metrics:                   # validation ALSO gets this
  confusion: !lazy:torchmetrics.classification.MulticlassConfusionMatrix
```

A same-named entry shadows, so a split can override one template rather than add to it
(`val_metrics: {accuracy_top1: !lazy:…MulticlassAccuracy {top_k: 2}}`). Merging needs both to be
mappings — a ready-made `MetricCollection` in a split slot *replaces*, since there is no key-wise
way to combine it.

**Every backend takes the slots its engine can serve**, and the asymmetry is the engine's, not an
omission:

| Backend | Slots | Why |
|---|---|---|
| lightning, torch, tensorflow | `train` · `val` · `test` | the loop is this project's, so each split gets its own metric objects |
| fastai, huggingface, jax, mlx | `val` · `test` | the engine's metric hook *is* its evaluation hook — a fastai `Learner` validates, a `transformers.Trainer` calls `compute_metrics` on eval, and the JAX / MLX training passes compute the loss and nothing else |
| keras | all three, **as their union** | `compile()` owns one metric list and reports it on train *and* validation, so a metric asked for on one split appears on the others |

Keras widening rather than dropping is deliberate: a metric computed where it was not asked for
is noise you can ignore, where a wired metric that silently reports nothing is a bug you cannot
see. `tests/test_metric_slots.py` pins the table in both directions — a backend must declare the
slots it serves, and must *not* offer one its engine cannot compute.

### Metrics that are not a single number (a confusion matrix)

Most metrics compute to one number, which is what a metric logger stores. A few do not — a
confusion matrix is `C × C` — and the example above is the reason the split slots exist: it
belongs on validation, not on a training split whose metrics are per-batch.

The scalars are logged exactly as before. The matrix takes the other route, because it cannot go
down the metric channel at all: a Lightning logger reduces every logged value to one number and
raises *"does not contain a single element"* on a matrix, and a `Tracker.log_metrics` call is
`{name: float}` in every backend. At the end of every validation and test epoch it is **rendered
as a table into the log, and pushed to each wired tracker's text channel** (`Tracker.log_text` —
MLflow artifact, TensorBoard's Text tab with a per-epoch step slider, Aim, ClearML):

```
val/confusion:

|        |   pred:0 |   pred:1 |   pred:2 | …
|--------|----------|----------|----------|
| true:0 |        8 |        0 |        0 | …
| true:1 |        0 |       14 |        0 | …
```

Rows are true classes, columns predicted (the torchmetrics convention). Labels come from the
run's `class_names` when the dataset has them, and are class indices otherwise. The metrics
sidecar and the `computed_metrics` output stay scalar-only.

Set `plot_metrics: true` to *also* send each matrix as a matplotlib heatmap through
`Tracker.log_figure` — glanceable, and the only readable form once there are many classes. It is
off by default because it makes matplotlib (`marainer[plot]`) a cost of every eval epoch, where
the table costs nothing and works in a plain log. See
[`examples/train_efficientvit_b0_mnist.yaml`](examples/train_efficientvit_b0_mnist.yaml) for the
wiring in context — and note that a backend on another engine must override both keys, since
torchmetrics classes are `framework="torch"`.

### Wiring a Keras loss or metric

The classifier's head is a bare `Dense` with **no activation**, so it emits raw logits. That
is one line of config on every crossentropy loss:

```yaml
keras: !scope:framework=keras
  runnable: !class:sonair.KerasClassifier
    loss: !class:keras.losses.SparseCategoricalCrossentropy
      from_logits: true            # NOT optional — see below
    metrics:
      - !class:keras.metrics.SparseCategoricalAccuracy
      - !class:keras.metrics.SparseTopKCategoricalAccuracy { k: 5 }
```

Keras's crossentropy losses default to `from_logits=False`, i.e. they expect probabilities.
Handed logits they normalize by the row **sum** instead of a softmax, then clip rather than
raise — so the loss stays finite and merely wrong, and the only symptom is a model that does
not learn. (This cost a real MNIST run: `loss: nan`, `val_accuracy: 0.098`.)

**Metrics: match the target shape first.** What the classifier hands a metric as `y_true`
decides which ones apply at all — the same split the torch backend spells as two families
(`Multiclass*` vs `Multilabel*`):

| Target | `y_true` shape | Metrics that read it |
|---|---|---|
| single-label (`Label`) | int64 ids `[N]` | `SparseCategoricalAccuracy`, `SparseTopKCategoricalAccuracy`, `SparseCategoricalCrossentropy@role=metric` |
| multi-label (`MultiLabel`) | float multi-hot `[N, C]` | `BinaryAccuracy`, `Precision`, `Recall`, `TruePositives` + the other three counters, `F1Score`, `FBetaScore`, `AUC`, `CategoricalAccuracy`, `TopKCategoricalAccuracy`, and the remaining `@role=metric` namesakes (`BinaryCrossentropy`, `CategoricalCrossentropy`, `Hinge`, `SquaredHinge`, `CategoricalHinge`, `KLDivergence`) |

Mixing them is loud, not subtle: `Precision` wired beside `SparseCategoricalAccuracy` on
10-class MNIST dies at the first batch with *"The size of tensor a (32) must match the size of
tensor b (320)"*.

**Then the logits knob**, for a shape-compatible metric:

| Kind | Metrics | On a logits head |
|---|---|---|
| argmax / top-k | the `*Accuracy` family, `F1Score` / `FBetaScore` with the default `threshold` | nothing to set — order-preserving, identical to the probability reading |
| thresholded | `Precision`, `Recall`, `BinaryAccuracy`, `TruePositives` + the other three counters | pass `thresholds: 0.0` (`threshold: 0.0` on `BinaryAccuracy`) — the logit-space equivalent of the probability default `0.5`, since `sigmoid(0) == 0.5` |
| `AUC` | — | pass `from_logits: true` |

`F1Score` / `FBetaScore` with an *explicit* `threshold` reject the logit-space value (Keras
requires `0 < threshold <= 1`), so either leave it unset or feed probabilities.

A wired `!class:` loss or metric is flowed at compile time, so both the plain and the
instance (`!class:X()`) spelling work; an explicitly wired one is never replaced by the
data-derived default.

### Seven names that are both a loss and a metric

Keras publishes `BinaryCrossentropy`, `CategoricalCrossentropy`,
`SparseCategoricalCrossentropy`, `Hinge`, `SquaredHinge`, `CategoricalHinge` and
`KLDivergence` under **both** `keras.losses` and `keras.metrics`. Both are registered, so a
bare reference to one of those names is ambiguous — say which role you mean:

```yaml
loss: !class:SparseCategoricalCrossentropy@role=loss,framework=keras
  from_logits: true
metrics:
  - !class:keras.metrics.SparseCategoricalAccuracy()             # dotted path
  - !class:SparseCategoricalCrossentropy@role=metric,framework=keras()   # the namesake
```

Tracking a loss you are *not* optimizing is the point — Keras already reports the compiled
one. Both spellings work everywhere: the dotted path is always unambiguous, and the
`@role=` selector is the readable form. Naming one bare fails with an error listing both
candidates rather than picking one.

Two upstream families are deliberately not registered, because they cannot consume what this
head emits and offer no knob to fix it: `keras.metrics.Accuracy` (wants already-argmaxed class
ids — reports `0.0` against logits) and `PrecisionAtRecall` / `RecallAtPrecision` /
`SensitivityAtSpecificity` / `SpecificityAtSensitivity` (sweep thresholds across `[0, 1]`).
That only keeps them out of the *pickers* — `!class:keras.metrics.Accuracy` still resolves
from YAML if your model really does emit class ids.

## Adapters

| Symbol | Purpose |
|--------|---------|
| `torch.nn.CrossEntropyLoss` | The default classification loss — registered (no sonair wrapper) as `task="classification", role="loss", framework="torch"`; set `label_smoothing` / `ignore_index` / `reduction` from YAML. Consumes int64 class-id targets `[N]`. |
| `SigmoidFocalLoss` | Thin `nn.Module` wrapper over `torchvision.ops.sigmoid_focal_loss` (a function, which confluid's `register` can't tag) — a pickable `task="classification", role="loss"` configurable with `alpha` / `gamma` / `reduction` knobs. Expects **float** targets the same shape as the logits (e.g. one-hot `[N, C]`), like the `BCEWithLogitsLoss` family. Defaults `reduction="mean"` (a scalar loss), not torchvision's `"none"`. The Keras side needs no such shim — `BinaryFocalCrossentropy` is already a class. |
| `torch.nn.BCEWithLogitsLoss` | The default **multi-label** loss (also registered, `role="loss"`). Consumes float multi-hot targets `[N, C]` shaped like the logits. |
| `recordstream.collate_records` | The generic DataLoader collate the classifier wires in (no bespoke sonair collate). Stacks the already-tensorized input under `input_key` (default `image`) to `[N, 3, H, W]` and leaves the other keys as per-record lists; the model boundary reads it back with `recordstream.batch_tensor` / `batch_metadata` and stacks the `Label` target under `target_key` (default `class`) into an `[N]` int64 tensor — or builds an `[N, C]` float multi-hot from a `MultiLabel`. |
| `load_lightning_checkpoint` | `marainer.torch` helper that strips Lightning's `"model."` prefix when loading a `.ckpt` saved by `sonair.LightningClassifier`. |

Most torchmetrics classification metrics accept logits directly and emit a
scalar — they wire into `val_metrics` straight from YAML, no adapter needed.
Both families are registered: `Multiclass*` (`MulticlassAccuracy`,
`MulticlassF1Score`, `MulticlassAUROC`, ...), sized by `num_classes`, and
`Multilabel*` (`MultilabelAccuracy`, `MultilabelF1Score`,
`MultilabelExactMatch`, ...), sized by `num_labels`. The classifier picks the
family that matches the data; name one explicitly to override. The Keras
equivalents — and the one knob they need against a logits head — are in
[Losses and metrics](#losses-and-metrics) above.

## Optional tracking via marainer

`LightningClassifier` is a standard `pytorch_lightning.LightningModule` that builds
its own `pytorch_lightning.Trainer` from the `lightning` slot / ergonomic knobs. It
uses marainer ONLY for OPTIONAL experiment tracking: when `trackers` are wired it
attaches a `marainer.lightning.TrackingCallback` (git/config/log-file provenance),
and `log_system_metrics: true` activates framework-free system-metric logging.
Removing marainer would take only a few line changes. There is no separate `logger:`
knob of `pytorch_lightning.loggers.*` — the trackers ARE this run's metric sinks, adapted
to Lightning's logger interface, so one key drives every backend.

### Pointing a tracker at a running service

Each tracking backend finds its server a different way, and a wrong guess fails quietly —
the run trains, the tracker records, and nothing appears in the UI you are watching:

```yaml
trackers:
  - !class:marainer.TensorBoardTracker      # a local directory, no service
    save_dir: ./lightning_logs

  - !class:marainer.AimTracker              # a local repository the UI serves
    save_dir: $AIM_REPO

  - !class:marainer.MlflowTracker           # a URL
    tracking_uri: http://localhost:5005

  - !class:marainer.ClearMLTracker          # nothing here at all — see below
    experiment_name: mnist_classification
```

`$VAR` is expanded when the app loads the config, so the variable must be **exported** in
the environment the run starts from. An unset one is left as the literal text `$AIM_REPO`
and becomes a directory of that name rather than an error. `ClearMLTracker` takes no host
key: the ClearML SDK reads `CLEARML_API_HOST` / `_WEB_HOST` / `_FILES_HOST` and its
credentials straight from the environment (`offline: true` records to local disk instead
and needs no server and no credentials).

`examples/train_efficientvit_b0_mnist.yaml` carries all four as commented alternatives.

### Framework callbacks

`trackers:` is a run's output wiring; `callbacks:` is its engine's own callback list — a
separate key because a callback is the one part of a run that is NOT engine-neutral:

```yaml
keras: !scope:framework=keras
  callbacks:
    - !class:keras.callbacks.EarlyStopping(patience=3)
    - !class:keras.callbacks.ModelCheckpoint(filepath=ckpt/best.keras)
```

Available on the `keras` / `tensorflow` / `fastai` / `huggingface` backends (their engine's
own `Callback` class) and on `jax` / `mlx`, whose loops are sonair's own and take a plain
`callback(epoch, logs)` callable. The `lightning` backend reaches the same thing through its
`lightning:` Trainer slot; the bare-`torch` backend has no callback protocol by design.

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
| `timm` | Underlying timm classification backbones (`[lightning]`, `[fastai]`). |
| `torch`, `torchmetrics`, `pytorch-lightning` | Standard PyTorch stack (`[lightning]`). |
| `keras` | Keras 3 — the API both `[keras]` and `[tensorflow]` wire to. |
| `tensorflow` | The engine under `[tensorflow]`, and what makes SavedModel export possible. |
| `fastai` | The `Learner` and the `*Flat` losses (`[fastai]`). |
