# Training backends

Sonair is a demonstration that **the config is the program**: the same
`runnable:` YAML, the same verbs, and the same logging backends hold while the
training engine underneath changes. The CLI makes that axis explicit.

Why it is shaped this way is recorded in
[Architecture Decisions](architecture.md).

```bash
sonair lightning  train    cfg.yaml    # implemented — PyTorch Lightning
sonair lightning  evaluate cfg.yaml
sonair lightning  predict  cfg.yaml

sonair keras      train  cfg.yaml     # implemented — Keras 3
sonair tensorflow train  cfg.yaml     # implemented — tf.data + tf.GradientTape
sonair fastai     train  cfg.yaml     # implemented — Learner + one-cycle
sonair huggingface train cfg.yaml     # implemented — transformers Trainer
sonair jax        train  cfg.yaml     # implemented — jitted optax loop
sonair mlx        train  cfg.yaml     # implemented — Apple-silicon unified memory
```

`sonair --help` lists the backends; `sonair <backend> --help` lists its verbs.
Every backend the CLI advertises is now implemented — the roadmap is finished.
Templates were how it advertised the unbuilt ones (the verbs existed and raised
an actionable error naming the file to create); `TEMPLATES` is empty today, and
the machinery stays because it is how the NEXT backend will be advertised.

## What each layer owns

| Layer | Varies per backend? | Where |
|---|---|---|
| CLI verbs (`train`/`evaluate`/`predict`) | no | [`cli.py`](../sonair/cli.py) |
| Config shape (the `runnable:` root) | no | the workspace-wide runner contract |
| Logging backends (`logger:`, `trackers:`) | no | wired by the runnable |
| Dataset / ops / collate | no | `recordstream` |
| **The runnable class** | **yes** | `sonair/<backend>_classifier.py` |
| **The model wrapper** | *sometimes* | `sonair/<backend>_model.py` |
| **Losses / metrics** | *sometimes* | `sonair/<framework>_classification.py` |

Only the last three rows are backend work — and the last of them is a list of
upstream class names, not code. That is the claim this project exists to
demonstrate, and it is why a template's error message can honestly say "the
CLI, the config shape and the logging backends are already wired".

Two of those rows say *sometimes* because backends differ in what they add.
`TensorFlowClassifier` reuses `KerasModel` and the Keras loss set (in TensorFlow
>= 2.16, `tf.keras` **is** Keras 3); `FastaiClassifier` reuses `TimmModel` and
the torch loss set (fastai runs on torch). `HuggingFaceClassifier` brings a model
wrapper but **no loss set at all**, because an `AutoModelForImageClassification`
computes its own loss from `config.problem_type` — there is nothing for a config
to wire. A backend adds a module only when it adds a genuinely new wiring
surface — see [Architecture Decisions §5](architecture.md).

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

**2. Move it from `TEMPLATES` to `IMPLEMENTED`** in
[`sonair/cli.py`](../sonair/cli.py). Both are tables of the three values a
backend's sub-app differs by, so promoting one is a two-line edit and no new
code:

```python
IMPLEMENTED: Dict[str, Backend] = {
    ...,
    "jax": Backend(
        "JAX + Flax/NNX — functional transforms, jit/pmap scaling.",
        "sonair.jax_classification",      # or an existing set, if it shares one
    ),
}
```

`_backend_app` builds the three `train` / `evaluate` / `predict` verbs from that
entry — each `flow_mode="manual"`, each importing the named registration module
and calling `_dispatch` with the verb's task. Declare the backend's dependencies
under the matching extra in `pyproject.toml` and install with
`pip install "sonair[jax]"`.

**3. Register its losses and metrics — if it needs a set of its own.** Two of the
four implemented backends do not: `tensorflow` names
`sonair.keras_classification` (TF >= 2.16's `tf.keras` IS Keras 3) and `fastai`'s
losses are torch callables, registered `framework="torch"`. When the engine's
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

There is still **no registry module or `Framework` dataclass** — the tables in
`cli.py` are data, not machinery. See [Architecture Decisions
§1](architecture.md) for what the fourth backend changed and what it did not.

## Adding a *new* framework

A new template is one line in `TEMPLATES`; a new working backend is a sub-app
like the one above.

Implemented: **PyTorch** (via Lightning — the reference), **Keras 3**,
**TensorFlow** (a native `tf.data` + `tf.GradientTape` loop) and **fastai**.
Registered as templates: **JAX** (Flax/NNX), **Hugging Face Transformers**
`Trainer` + Accelerate, and **MLX** (Apple silicon; its extra carries a platform
marker so `pip install "sonair[mlx]"` stays installable elsewhere).

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
train_set: !class:recordstream.sources.HuggingFaceSource
  path: mnist
  split: train
batch_size: 32
max_epochs: 3

# framework-specific — top level is the default (lightning)
runnable: !class:sonair.LightningClassifier
model: !lazy:sonair.TimmModel
  model_name: efficientvit_b0

# an alternative overrides only those keys
keras: !scope:framework=keras
  runnable: !class:sonair.KerasClassifier
  model: !lazy:sonair.KerasModel
    model_name: efficientvit_b0
```

The `sonair <backend>` verb activates the matching scope for you, so the backend
is never named twice:

```bash
sonair keras train cfg.yaml                         # implies --scope framework=keras
recordstream run  cfg.yaml --scope framework=keras  # the explicit form
recordstream run  cfg.yaml                          # no scope -> the default backend
```

An explicit `--scope framework=…` or `--framework …` always wins over the verb.

[`examples/train_efficientvit_b0_mnist.yaml`](../examples/train_efficientvit_b0_mnist.yaml)
is built this way. Its alternative blocks are **commented out** while their
runnable classes do not exist — uncommenting one both activates that backend and
declares the `framework` dimension, which is what makes the implicit
`--framework <name>` flag bind.

## Why templates instead of hiding unimplemented backends

A template fails with the file to create, the reference to copy, and the exact
registry line that switches it on:

```console
$ sonair jax train
Error: `sonair jax train` is not implemented yet — jax is a backend TEMPLATE.
...
  1. Create sonair/jax_classifier.py with a runnable class: ...
  2. Move `jax` from TEMPLATES to IMPLEMENTED in sonair/cli.py, naming the
     module that registers its losses/metrics.
  3. Install the backend: pip install 'sonair[jax]'
```

That is exactly how the `keras` backend got built (2026-07-29): the template named the
file, the reference and the edit, and graduating it meant deleting its entry from
`TEMPLATES`, adding a real sub-app, and moving it to `IMPLEMENTED` in
`tests/test_cli_frameworks.py` — which then holds it to the same verb surface as the rest.

It exits 1 through liquifai's failure contract (one clean message, full
traceback under `--debug`) rather than raising an unhandled `NotImplementedError`
— an unbuilt backend is a documented state, not a crash.
