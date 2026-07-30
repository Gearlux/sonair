# Training backends

Sonair is a demonstration that **the config is the program**: the same
`runnable:` YAML, the same verbs, and the same logging backends hold while the
training engine underneath changes. The CLI makes that axis explicit.

Why it is shaped this way — and why the registry has no `framework` axis yet —
is recorded in [Architecture Decisions](architecture.md).

```bash
sonair lightning train    cfg.yaml     # implemented
sonair lightning evaluate cfg.yaml
sonair lightning predict  cfg.yaml

sonair keras       train  cfg.yaml    # implemented (Keras 3)
sonair tensorflow  train  cfg.yaml    # template
sonair jax         train  cfg.yaml    # template
sonair huggingface train  cfg.yaml    # template
sonair mlx         train  cfg.yaml    # template
```

`sonair --help` lists the backends; `sonair <backend> --help` lists its verbs.
Templates advertise the same three verbs on purpose — the roadmap should be
visible from the CLI, not buried in a tracker.

## What each layer owns

| Layer | Varies per backend? | Where |
|---|---|---|
| CLI verbs (`train`/`evaluate`/`predict`) | no | [`cli.py`](../sonair/cli.py) |
| Config shape (the `runnable:` root) | no | the workspace-wide runner contract |
| Logging backends (`logger:`, `trackers:`) | no | wired by the runnable |
| Dataset / ops / collate | no | `recordstream` |
| **The runnable class** | **yes** | `sonair/<backend>_classifier.py` |
| **Losses / metrics** | **yes** | `sonair/<framework>_classification.py` |

Only the last two rows are backend work — and the second of them is a list of
upstream class names, not code. That is the claim this project exists to
demonstrate, and it is why a template's error message can honestly say "the
CLI, the config shape and the logging backends are already wired".

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

**2. Replace its template block** in [`sonair/cli.py`](../sonair/cli.py) with a
real sub-app, modelled on `lightning` right above it — drop the name from
`TEMPLATES` and write the three verbs out:

```python
keras = LiquifyApp(name="keras", description="Keras 3 — one API over TF / JAX / PyTorch.")

@keras.script_command(name="train", flow_mode="manual")
def train(runnable: Any) -> None:
    """Train an image classifier with Keras (task=fit)."""
    _dispatch(runnable, "fit")

# ... evaluate / predict likewise

app.add_app(keras, "keras")
```

Declare the backend's dependencies under the matching extra in `pyproject.toml`
and install with `pip install "sonair[keras]"`.

**3. Register its losses and metrics.** Create
`sonair/<framework>_classification.py` — two tuples of *upstream* classes, a
`register(...)` loop each, and a literal `__all__` naming every one — then add it
to `[project.entry-points."confluid.configurables"]`:

```python
from confluid import register

_CLASSIFICATION_LOSSES = (SparseCategoricalCrossentropy, BinaryCrossentropy, ...)
for _loss_cls in _CLASSIFICATION_LOSSES:
    register(_loss_cls, task="classification", role="loss", framework="keras")
```

Two constraints worth knowing before you write the tuples. The registry is keyed
by class **name**, so a name published under both the loss and metric namespaces
(Keras has thirteen: `BinaryCrossentropy`, `Hinge`, `KLDivergence`, …) can be
registered only once — register the loss, skip the metric duplicate. And a metric
that cannot consume what the classifier's head emits does not belong in a picker:
sonair's head is a bare linear layer, so `keras.metrics.Accuracy` (which wants
already-argmaxed ids) and the `PrecisionAtRecall` family (which sweeps thresholds
across `[0, 1]`) are left out. Leaving a class unregistered removes it from the
pickers only — `!class:keras.metrics.Accuracy` still resolves from YAML.

There is deliberately **no registry module or `Framework` dataclass** — see
[Architecture Decisions §1](architecture.md) for why, and for the condition
(a third implemented backend) that would justify adding one.

## Adding a *new* framework

A new template is one line in `TEMPLATES`; a new working backend is a sub-app
like the one above.

Registered so far: **PyTorch** (via Lightning — the working one), **Keras 3**,
**TensorFlow**, **JAX** (Flax/NNX), **Hugging Face Transformers** `Trainer` +
Accelerate, and **MLX** (Apple silicon; its extra carries a platform marker so
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
$ sonair tensorflow train
Error: `sonair tensorflow train` is not implemented yet — tensorflow is a backend TEMPLATE.
...
  1. Create sonair/tensorflow_classifier.py with a runnable class: ...
  2. Replace the `tensorflow` template in sonair/cli.py with a real sub-app,
     modelled on the `lightning` one above it.
  3. Install the backend: pip install 'sonair[tensorflow]'
```

That is exactly how the `keras` backend got built (2026-07-29): the template named the
file, the reference and the edit, and graduating it meant deleting its entry from
`TEMPLATES`, adding a real sub-app, and moving it to `IMPLEMENTED` in
`tests/test_cli_frameworks.py` — which then holds it to the same verb surface as the rest.

It exits 1 through liquifai's failure contract (one clean message, full
traceback under `--debug`) rather than raising an unhandled `NotImplementedError`
— an unbuilt backend is a documented state, not a crash.
