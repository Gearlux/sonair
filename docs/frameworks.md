# Training backends

Sonair is a demonstration that **the config is the program**: the same
`runnable:` YAML, the same verbs, and the same logging backends hold while the
training engine underneath changes. The CLI makes that axis explicit.

Why it is shaped this way is recorded in
[Architecture Decisions](architecture.md).

```bash
sonair train    cfg.yaml                       # the config's default backend
sonair evaluate cfg.yaml
sonair predict  cfg.yaml

sonair train cfg.yaml --framework torch        # a hand-written loop, no framework
sonair train cfg.yaml --framework keras        # Keras 3
sonair train cfg.yaml --framework tensorflow   # tf.data + tf.GradientTape
sonair train cfg.yaml --framework fastai       # Learner + one-cycle
sonair train cfg.yaml --framework huggingface  # transformers Trainer
sonair train cfg.yaml --framework jax          # jitted optax loop
sonair train cfg.yaml --framework mlx          # Apple-silicon unified memory
```

`sonair frameworks` lists them with a line each. The same flag drives the generic
runner — `recordstream run cfg.yaml --framework keras` — because it is not really
a CLI feature at all: it activates a `!scope:framework=<name>` block in the config,
and any runner that loads a confluid document can do that.

The name must match a block the config declares. `--framework kears` is an error
listing the backends that file *does* declare, rather than a silent fall-through to
the default — so a mistyped or unsupported backend fails immediately instead of
training the wrong thing.

Every backend the CLI advertises is implemented — the roadmap is finished. An
unbuilt one is simply absent from `FRAMEWORKS`; there is no stub to mount now that
the backend is a flag rather than a subcommand.

## What each layer owns

| Layer | Varies per backend? | Where |
|---|---|---|
| CLI verbs (`train`/`evaluate`/`predict`) | no | [`cli.py`](../sonair/cli.py) |
| Config shape (the `runnable:` root) | no | the workspace-wide runner contract |
| Tracking backends (`trackers:`) | no | wired by the runnable |
| Dataset / ops / collate | no | `recordstream` |
| **The runnable class** | **yes** | `sonair/<backend>_classifier.py` |
| **The model wrapper** | *sometimes* | `sonair/<backend>_model.py` |
| **A shared-spec convnet** | *sometimes* | `sonair/convnet/<backend>_convnet.py` |
| **Losses / metrics** | *sometimes* | `sonair/<framework>_classification.py` |

Only the last four rows are backend work — and the last of them is a list of
upstream class names, not code. That is the claim this project exists to
demonstrate, and it is why a template's error message can honestly say "the
CLI, the config shape and the logging backends are already wired".

Three of those rows say *sometimes* because backends differ in what they add.
`TensorFlowClassifier` reuses the Keras loss set — and can take `KerasModel`
itself — because in TensorFlow >= 2.16, `tf.keras` **is** Keras 3;
`FastaiClassifier` and `TorchClassifier` reuse `TimmModel` and the torch loss
set (both run on torch, and `TorchClassifier` runs on nothing else). `HuggingFaceClassifier` brings a model wrapper
but **no loss set at all**, because an `AutoModelForImageClassification` computes
its own loss from `config.problem_type` — there is nothing for a config to wire.
A backend adds a module only when it adds a genuinely new wiring surface — see
[Architecture Decisions §5](architecture.md).

The convnet row is separate from the model-wrapper row because the two wrap
different things: a model wrapper reaches somebody else's weights, while a
convnet implements the [shared conv spec](../sonair/convnet/spec.py) and wraps
nothing at all. That is why the convnets live together in
[`sonair/convnet/`](../sonair/convnet/) rather than beside their backend's zoo
wrapper — the four are the same network in four idioms, so the directory listing
*is* the comparison:

```
sonair/convnet/
  spec.py                 the architecture, written once
  torch_convnet.py        TorchConvNet        torch.nn      (lightning AND fastai)
  keras_convnet.py        KerasConvNet        keras.layers
  tensorflow_convnet.py   TensorFlowConvNet   raw tf.nn ops on a tf.Module
  flax_convnet.py         FlaxConvNet         flax.nnx
  mlx_convnet.py          MlxConvNet          mlx.nn
```

Each module is named for the class it exports, exactly as at the top level;
`spec.py` is the exception because it exports no class.

Each convnet **is** its framework's own model type — a `torch.nn.Module`, a
`keras.Model`, a `tf.Module`, an `nnx.Module`, an `mlx.nn.Module` — just as
`TimmModel` is a `torch.nn.Module`. So a classifier drives whatever is wired
into its `model` slot without reaching through a wrapper to find the network,
and the `model` slot of each backend is typed as that framework's module rather
than as the concrete sonair class.

**Eight backends, five convnets, and the arithmetic is not a gap.** Three
backends run on torch — `lightning`, `torch` and `fastai` — and all three take
`TorchConvNet` unchanged, the same reason they already share `TimmModel`, and why
there is no `fastai_convnet.py`. `tensorflow` can take `KerasConvNet` too, since
`tf.keras` IS Keras 3. The one backend with no convnet is **`huggingface`**, and that is structural rather than
unfinished: `transformers.Trainer` calls the model with `labels=` and reads
`outputs["loss"]` back, so it drives models that compute their own loss from
`config.problem_type`. A bare convnet returning logits does not satisfy that
contract, and making it do so would mean writing a `PreTrainedModel` subclass
with a `transformers` config — i.e. wrapping the convnet in the very zoo
machinery the spec exists to avoid. Wire `HuggingFaceModel` there instead.

A backend that brings both kinds brings both files — `lightning` has
`timm_model.py` and `convnet/torch_convnet.py`, `keras` has `keras_model.py` and
`convnet/keras_convnet.py`, `tensorflow` has `tensorflow_model.py` and
`convnet/tensorflow_convnet.py`. The `jax` and `mlx` backends have no zoo to
wrap, so they bring only a convnet; `huggingface` has no convnet, so it brings
only a zoo wrapper.

## The verb → task mapping

| Verb | `task` | Meaning |
|---|---|---|
| `train` | `fit` | train, then test if a `test_set` is wired |
| `evaluate` | `test` | score a trained model over the held-out split |
| `predict` | `predict` | stream predictions to the configured sink |

The **verb wins** over any `task:` in the YAML, so one config serves all three.
`recordstream run cfg.yaml` has no verb, so it uses whatever `task:` the YAML
declares — or the runnable's constructor default (`fit`) when the config omits
it, which is why the shipped example carries no `task:` line at all. Set `task:`
only to change what a bare `recordstream run` does.

## Adding a backend

Three steps, no plumbing.

> **Losses and metrics are engine-bound, so each engine registers its own set
> from its own module.**
> [`torch_classification.py`](../sonair/torch_classification.py) registers the
> `torch.nn` / `torchmetrics` classes as `framework="torch"`;
> [`keras_classification.py`](../sonair/keras_classification.py) registers the
> `keras.losses` / `keras.metrics` classes as `framework="keras"`. A picker asked
> for a classification loss for a Keras trainer is offered only the Keras set —
> that filtering is real as of confluid 0.3.0 ([Architecture Decisions
> §2](architecture.md)). A new backend adds `<framework>_classification.py` and
> one entry-point line; a module whose engine is not installed simply fails to
> import and is skipped.

**1. Write the runnable.** Create `sonair/<name>_classifier.py` with a class that:

- is `@configurable(task="classification", role="trainer")` so navigaitor and
  StreamStudio discover it;
- takes a `task` parameter typed as a closed `Literal`;
- has `run()` → `initialize()` → dispatch to `@entrypoint`-tagged methods;
- does no functional work in `__init__` (build the model in `initialize()`).

[`sonair/lightning_classifier.py`](../sonair/lightning_classifier.py) is the
reference implementation; `recordstream/docs/runnable.md` documents the
runnable protocol itself.

**2. Add a row to `FRAMEWORKS`** in [`sonair/cli.py`](../sonair/cli.py). It is a
table of the two values a backend differs by — what to say about it, and which
module registers its losses/metrics — so this is a one-entry edit and no new code:

```python
FRAMEWORKS: Dict[str, Framework] = {
    ...,
    "jax": Framework(
        "JAX + Flax/NNX — functional transforms, jit/pmap scaling.",
        "sonair.jax_classification",      # or an existing set, if it shares one
    ),
}
```

The three verbs are built once, for every backend, so a new row adds no commands.
What the row drives is `sonair frameworks` and — when `--framework jax` is passed
— importing that registration module before the config is loaded, so a config may
name the engine's losses by their bare names. Declare the backend's dependencies
under the matching extra in `pyproject.toml` and install with
`pip install "sonair[jax]"`.

**3. Register its losses and metrics — if it needs a set of its own.** Three
modules serve eight backends, so five of them add nothing here. `torch` shares
`sonair.torch_classification` with `lightning` (a `torch.nn` loss does not care
which loop calls it), `tensorflow` names `sonair.keras_classification`
(TF >= 2.16's `tf.keras` IS Keras 3), and `fastai`'s losses are torch callables
registered `framework="torch"`. Three name **no** module at all
(`Backend.registrations is None`): `huggingface`, because an
`AutoModelFor*` computes its own loss from `config.problem_type` and there is
nothing a config could wire; `jax` and `mlx`, because optax and `mlx.nn.losses`
ship plain FUNCTIONS, which confluid's `register` — a class decorator — cannot
tag. Those two resolve a loss by NAME instead. (`Framework.registrations` is what
this refers to.) When the engine's
loss API *is* new, create `sonair/<framework>_classification.py` — tuples of
*upstream* classes, a `register(...)` loop each, and a literal `__all__` naming
every one — then add it to
`[project.entry-points."confluid.configurables"]`:

```python
from confluid import register

_CLASSIFICATION_LOSSES = (SparseCategoricalCrossentropy, BinaryCrossentropy, ...)
for _loss_cls in _CLASSIFICATION_LOSSES:
    register(_loss_cls, task="classification", role="loss", framework="keras")
```

Two constraints worth knowing before you write the tuples. The registry is keyed
by class **name**, so a name published under both the loss and metric namespaces
(Keras has thirteen: `BinaryCrossentropy`, `Hinge`, `KLDivergence`, …) needs a tag
that tells the two entries apart — `role` does, so both are registered and a
config picks one with `!class:Hinge@role=metric`. And a metric that cannot consume
what the classifier's head emits does not belong in a picker: sonair's head is a
bare linear layer, so `keras.metrics.Accuracy` (which wants already-argmaxed ids)
and the `PrecisionAtRecall` family (which sweeps thresholds across `[0, 1]`) are
left out. Leaving a class unregistered removes it from the pickers only —
`!class:keras.metrics.Accuracy` still resolves from YAML.

There is still **no registry module** — `FRAMEWORKS` is data, not machinery, and
`Framework` is a two-field `NamedTuple`. See [Architecture Decisions
§1](architecture.md) for what the fourth backend changed, and §1b for why the
eighth one collapsed the sub-apps into a flag.

## The backends today

All eight are implemented: **PyTorch** (via Lightning — the reference), **bare
PyTorch** (a hand-written loop, no training framework at all), **Keras 3**,
**TensorFlow** (a native `tf.data` + `tf.GradientTape` loop), **fastai**,
**Hugging Face Transformers** (`Trainer` + Accelerate), **JAX** (Flax/NNX) and
**MLX** (Apple silicon; its extra carries a platform marker so
`pip install "sonair[mlx]"` stays installable elsewhere).

Still unregistered — for non-deep-learning baselines,
**scikit-learn** and the gradient-boosting family (**XGBoost** / **LightGBM** /
**CatBoost**) would exercise the same contract against a very different engine —
a useful stress test of the claim that the config shape is engine-agnostic.

## One config, every backend

The config shape is engine-agnostic, so a single file carries all of them:
shared keys at the top level (which is also the **default** backend) and one
`!scope:framework=<name>` block per alternative, overriding *only* the
framework-specific keys. Confluid splices an active scope block into its parent
in document order, last write wins — so the block overrides what the top level
set.

```yaml
# shared — identical for every backend
train_set: !class:recordstream.sources.huggingface.HuggingFaceSource
  path: mnist
  split: train
batch_size: 32
max_epochs: 3

# framework-specific — top level is the default (lightning)
runnable: !class:sonair.LightningClassifier
model: !lazy:sonair.TimmModel
  model_name: efficientvit_b0

# the default, named — so `--framework lightning` is a checkable value, not a no-op
lightning: !scope:framework=lightning
  runnable: !class:sonair.LightningClassifier

# an alternative overrides only those keys
keras: !scope:framework=keras
  runnable: !class:sonair.KerasClassifier
  model: !lazy:sonair.KerasModel
    model_name: efficientvit_b0
```

Note that **every framework-specific key is a top-level sibling** — `runnable`,
`model`, `trackers` — rather than `model:` living inside `runnable:`. That is not
style. A scope block replaces a *key*, taking its whole subtree with it, so a
nested `model:` disappears the instant a block swaps the runnable, and the
backend that inherits it dies with `__init__() missing 1 required positional
argument: 'model'`. Keep the slots siblings and a block can override one without
erasing another.

The flip side of sharing the top level is that an inherited key is only right
for the backends whose engine can consume it. **The metric keys are engine-bound**
— the example's `metrics:` and `val_metrics:` are `torchmetrics` classes, which
the three torch-backed backends read unchanged and `keras.Model.compile()` cannot
take at all — so a block on another engine must override both. Writing
`metrics: null` restores the classifier's own sentinel, which is *not* the same
as an empty list: `None` means "not chosen", so `initialize()` resolves the
engine's defaults from the data. (`val_metrics:` is doubly inapplicable there —
only the two torch backends have per-split metric slots at all; a Keras
`compile()` takes one metric list for the whole run.) The same applies
to any key whose value is a framework object (`loss`, `optimizer`). Tracking is
NOT among them: a marainer tracker is engine-neutral, so `trackers:` is shared by
every block rather than overridden per framework.
`tests/test_cli_frameworks.py::test_no_framework_inherits_a_metric_set_its_engine_cannot_consume`
pins it at the config level, so it fails wherever the mistake is made rather
than only on a machine that has the engine installed.

Activate a block with `--framework`, on either runner:

```bash
sonair train      cfg.yaml --framework keras        # the shorthand
recordstream run  cfg.yaml --framework keras        # same flag, generic runner
recordstream run  cfg.yaml --scope framework=keras  # what both expand to
recordstream run  cfg.yaml                          # no scope -> the default backend
```

`--framework <name>` must match a block. Confluid raises on an activation that
matches nothing, listing the values the document declares:

```console
$ sonair train cfg.yaml --framework kears
Error: No scope block matches framework='kears'. This document declares framework
with: keras, lightning. Either use one of those values, or add a
`!scope:framework=kears` block.
```

This is why the default backend gets a block of its own. The top level already
*is* lightning, so the block only restates the runnable — but without it,
`--framework lightning` would name a value nothing declares and be rejected.

[`examples/train_efficientvit_b0_mnist.yaml`](../examples/train_efficientvit_b0_mnist.yaml)
is built this way. Its `torch` and `keras` blocks are **live**; the rest stay
**commented out** — not because their runnable classes are missing (every backend
is implemented) but because their extras are not in the workspace's selected
frameworks, and loading the config imports every `!class:` it names, so a live
block would break `tests/test_examples_load.py` on a default checkout. Select the
extra (`pip install 'sonair[tensorflow]'`) and uncomment the block you want:
uncommenting one both activates that backend and declares the `framework`
dimension, which is what makes the implicit `--framework <name>` flag bind.

## Adding a backend

Three steps, and only the first is real work:

1. Write `sonair/<name>_classifier.py` — a runnable class with a `task` Literal,
   `run()` → `initialize()` → `@entrypoint`-tagged methods, and
   `@configurable(task="classification", role="trainer", framework=…)`. Copy
   [`lightning_classifier.py`](../sonair/lightning_classifier.py).
2. Add a row to `FRAMEWORKS` in [`cli.py`](../sonair/cli.py) — a description and
   the module that registers its losses/metrics (or `None`) — plus the name in
   `EXPECTED_FRAMEWORKS` in `tests/test_cli_frameworks.py`, a `[<name>]` extra in
   `pyproject.toml`, and a `confluid.configurables` entry point for the new module.
3. Add a `!scope:framework=<name>` block to a config.

Until 2026-08-01 an unbuilt backend was *advertised*: a `TEMPLATES` dict mounted
its verbs so `--help` showed the roadmap, and calling one raised an actionable
error naming the file to create. That machinery went with the sub-apps, since a
flag has nowhere to hang a stub — `--framework <name>` for a name that is not in
`FRAMEWORKS` simply imports no registrations, and the config's own scope check
reports what it does declare. `sonair frameworks` now lists exactly what is real,
which is a smaller and more honest claim than a menu of half of them.
