# Architecture Decisions

Why sonair is shaped the way it is. Each record answers one "why does this
exist?" question as **Context → Decision → Consequences → Example → What you may
change**.

This file is design *rationale*. How to USE a mechanism belongs in the user
docs — [training backends](frameworks.md) and the [README](../README.md).

---

## 1. Backends are sub-apps written out, not a registry

*2026-07-28*

**Context.** Sonair exposes the training engine as a CLI dimension
(`sonair lightning train`), with one working backend and five templates. The
first implementation put that behind a registry: a `Framework` dataclass with
`implemented` / `target_module` / `install_hint` properties, a `FRAMEWORKS`
tuple, a `BY_NAME` index, and a `frameworks.py` module — from which `cli.py`
generated every sub-app.

That is machinery sized for a problem sonair does not have. One backend works;
the rest are stubs that share a single failure path. The indirection made the
*working* backend harder to read than the plain three decorated functions it
replaced, for no behaviour that the plain version lacked.

**Decision.** No registry. `cli.py` contains everything:

- `lightning` is written out plainly — three `@script_command` functions calling
  `_dispatch`, reading like any hand-written liquifai CLI;
- the not-yet-built backends are a `TEMPLATES` dict plus a small
  `_template_app` factory, because five *identical* stubs genuinely are a loop.

**Consequences.**

- The working backend is greppable: `sonair lightning train` leads straight to a
  function named `train` with a docstring, not to a factory that synthesizes one.
- Adding a *template* is one dict entry; promoting one to real means writing a
  sub-app and deleting its entry — a visible, local edit.
- The asymmetry (one written out, five looped) is deliberate. It mirrors the
  actual asymmetry in the codebase, rather than hiding it behind uniform
  generation.
- If a **third real backend** lands and the sub-apps start duplicating logic,
  factor it out *then* — with three concrete cases to generalise from instead of
  one plus guesses.

**Example.**

```python
@lightning.script_command(flow_mode="auto")
def train(runnable: Any) -> None:
    """Train an image classifier with PyTorch Lightning (task=fit)."""
    _dispatch(runnable, "fit")

TEMPLATES = {"keras": "Keras 3 — one API over TF / JAX / PyTorch. TEMPLATE.", ...}
```

**What you may change.** Add templates freely. Do not reintroduce a registry
module for fewer than three implemented backends; when you do, generalise from
what the real sub-apps have in common, not from what a dataclass could express.

### 1a. …and the amendment the fourth backend triggered

*2026-07-30*

`tensorflow` and `fastai` were the third and fourth implemented backends, so the
condition the record above set was met. What twelve near-identical handlers had
in common turned out to be exactly three values — the name, the description, and
which loss/metric module to import — so the generalisation is a table of those
three plus a factory, and stops there:

```python
class Backend(NamedTuple):
    description: str
    registrations: str          # the module whose import registers this engine's set

IMPLEMENTED: Dict[str, Backend] = {
    "lightning": Backend("PyTorch Lightning — the reference backend …", "sonair.torch_classification"),
    "tensorflow": Backend("TensorFlow — a native tf.data + tf.GradientTape loop …",
                          "sonair.keras_classification"),   # tf.keras IS Keras 3
    ...
}

def _backend_app(backend: str, spec: Backend) -> LiquifyApp:
    sub = LiquifyApp(name=backend, description=spec.description)
    for verb in VERB_TASKS:
        sub.script_command(name=verb, flow_mode="manual")(_backend_command(verb, backend, spec))
    return sub
```

Note what stayed OUT of the table. `sonair tensorflow` must pin
`KERAS_BACKEND=tensorflow` before anything imports keras — earlier than any
handler can run, since liquifai resolves the config (and therefore imports the
runnable) first. That is a genuine asymmetry between the backends, so it lives
visibly in `main()` as `_pin_keras_backend` rather than as a fifth column that
would be `None` three times out of four.

The predictions the original record made held up: the generalisation came from
three real cases, and it is still data plus one factory rather than a `Framework`
dataclass with `implemented` / `target_module` / `install_hint` properties.

**What you may change.** Add rows to `IMPLEMENTED` / `TEMPLATES`. Do not grow
`Backend` to carry per-backend *behaviour* — a field that only one backend uses
is the signal that the behaviour belongs in the one place that needs it.

---

## 2. The framework axis: an explicit tag, with MRO inference as fallback

*2026-07-28, implemented 2026-07-29*

**Context.** Sonair registers off-the-shelf losses and metrics rather than
wrapping them — `torch.nn.CrossEntropyLoss` and the
`torchmetrics.classification.Multiclass*` family. They were tagged
`task="classification", role="loss"` / `role="metric"`, which say **what a class
is for** but never **which engine it runs on**.

With one backend that distinction is invisible. With a second it becomes
wrong-by-construction: a picker asked for "a classification loss" would offer
`torch.nn.CrossEntropyLoss` to a Keras trainer, and the resulting config fails at
run time with a type error far from its cause.

**Decision.** Add `framework` to confluid as a **third orthogonal axis**,
alongside `task` and `role` — explicit tag first, MRO inference as a consumer-side
fallback.

The unit is the **API, not the tensor runtime**: a `keras.losses.Loss` is
`"keras"` whether Keras runs on TensorFlow, JAX or PyTorch, because the API is
what decides whether a trainer can consume it. (This was the open question when
the decision was first recorded; it is what the "wire-to" framing settles.)

**Consequences.**

- confluid 0.3.0 stamps `__confluid_framework__`, indexes `_by_framework`, and
  adds `framework=` to `list_classes`'s intersecting filters. It is deliberately
  **not** folded into `category`, which stays `f"{task}_{role}"`. Consumers must
  raise their floor to `confluid>=0.3.0` — passing the kwarg against 0.2.0 is a
  `TypeError` at decoration time, i.e. at import.
- **Explicit, not inferred**, for the same reason `role` is: a `register()`-ed
  builder *function* (raidar's torchvision `fasterrcnn_*`) has no MRO to read.
  Inference exists only as a consumer-side fallback for untagged classes
  (`generate._framework_of` / `_TYPE_FRAMEWORK_BY_BASE`), mirroring StreamStudio's
  long-standing `_role_for_class` precedent.
- **Exclusion requires two claims.** A candidate that declares no framework is
  never filtered out, and a runnable that declares none filters nothing. Every
  untagged project (raidar, deltaid, waivefront) therefore behaves exactly as
  before — the axis is opt-in per class, not a flag day.
- **Framework is enforcement, task is guidance.** navigaitor deliberately offers
  cross-*task* options (a segmentation model in a classification model slot — "a
  model is a model"), ranked after the preferred tier. Cross-*framework* options
  are not offered at all, because they are not a choice: they cannot run.
- **StreamStudio is deliberately out of scope.** Its object sockets are
  string-equality-enforced by ComfyUI (`RECORDSTREAM_OBJECT:model`) and the type
  strings are persisted in saved `workflow.json` files, so qualifying them by
  framework would break every existing graph. That is a workflow-format change
  deserving its own decision, not a rider on this one. See `TASKS.md`.
- **One registration module per engine** — `torch_classification.py` (2026-07-30,
  renamed from `classification.py`, which read as task-scoped in a project whose
  whole point is that the task outlives the engine) and `keras_classification.py`.
  The tag alone would not have been enough: each module is its own
  `confluid.configurables` entry point, so on a torch-only install the Keras
  module fails to import and the scanner skips it — the registrations are absent
  because the engine is, which is the honest answer rather than a set of names
  that cannot be built. Splitting also keeps the import cost paid only by whoever
  installed that backend.
- **The registry is keyed by class NAME, which curates the set for you.** Keras
  publishes thirteen names under both `keras.losses` and `keras.metrics`
  (`BinaryCrossentropy`, `Hinge`, `KLDivergence`, …); registering both spellings
  would silently overwrite the loss with the metric. So the Keras metric set omits
  the duplicates. A second, independent curation applies for the same reason
  framework-filtering does: `keras.metrics.Accuracy` wants already-argmaxed class
  ids and the `PrecisionAtRecall` family sweeps thresholds across `[0, 1]` with no
  `from_logits` knob, so against this project's raw-logits head they report a
  wrong number rather than failing — and an option that cannot work is not an
  option. Neither exclusion blocks anything: registration governs DISCOVERY, and
  `!class:keras.metrics.Accuracy` still resolves from YAML.

**Example.**

```python
# tag at registration — classes and builder functions alike, one module per engine
register(nn.CrossEntropyLoss, task="classification", role="loss", framework="torch")           # torch_classification.py
register(keras.losses.BinaryCrossentropy, task="classification", role="loss", framework="keras")  # keras_classification.py

# the picker query that is now answerable
list_classes(task="classification", role="loss", framework="keras")   # -> no torch losses
```

**What you may change.** Tag a class's framework whenever it is engine-bound;
leave it off when it genuinely is not (a pure-python op, a path-based source).
Add an engine by adding a `<framework>_classification.py` + its entry point, not
by widening an existing module. Do not fold framework into `category`, and do not
make an untagged class excluded-by-default — the opt-in asymmetry is what let this
ship without re-tagging the workspace.

---

## 3. Task defaults are sentinels, resolved at `initialize()`

*2026-07-29*

**Context.** `LightningClassifier` has always shipped a working default loss
and metric set, so a minimal config trains without naming either. The obvious
spelling is a live default in the constructor signature:

```python
def __init__(self, loss_fn: Any = nn.CrossEntropyLoss(), ...):
```

Multi-label support broke that. A `MultiLabel` target needs a *different* loss
(`BCEWithLogitsLoss` over a float multi-hot `[N, C]`, not `CrossEntropyLoss`
over int64 ids `[N]`) and a *different* metric family (`Multilabel*` sized by
`num_labels`, not `Multiclass*` sized by `num_classes`). Which one is right is
a property of the DATA, and the data is not read until `initialize()` — long
after the constructor ran.

A live default also destroys the information needed to fix this later: once
`self.loss_fn` holds a `CrossEntropyLoss`, nothing can tell whether the user
asked for it or merely didn't say. Overwriting it when the data turns out to be
multi-label would silently discard a deliberate choice; leaving it would
silently train a multi-label model with the wrong loss.

**Decision.** `loss_fn` and `metrics` default to **`None`** — a sentinel
meaning *not chosen*, not *no loss*. `_resolve_task_defaults()` fills a `None`
slot the moment `initialize()` has set `self.multilabel` from the first target,
and is a no-op for anything already set. `_metric_dims()` is the matching
one-liner that returns `{"num_labels": n}` or `{"num_classes": n}` for
`build_metric_collection`.

**Consequences.**

- A wired loss/metric is never replaced, whatever the data looks like.
- The defaults are correct for both target shapes with no config knob and no
  `multilabel: true` flag to keep in sync with the dataset.
- The slots are `None` between construction and `initialize()`. Any code
  reading them earlier sees the sentinel — the step functions therefore call
  `_resolve_task_defaults()` themselves, so a directly-invoked
  `training_step` (tests, a custom loop) still works.
- Resolution ORDER matters inside `initialize()`: the class-balancing probe
  asks `self.accepts_class_weights()`, and an unresolved `None` loss answers
  "cannot be balanced" — so resolution happens *before* the probe, not after.

**Example.**

```python
def _resolve_task_defaults(self) -> None:
    """Fill the unset (None) task-shaped slots. Idempotent; never overwrites."""
    if self.loss_fn is None:
        self.loss_fn = nn.BCEWithLogitsLoss() if self.multilabel else nn.CrossEntropyLoss()
    if self.metrics is None:
        self.metrics = (
            {"accuracy": LazyClass[Metric](MultilabelAccuracy), ...}
            if self.multilabel
            else {"accuracy_top1": LazyClass[Metric](MulticlassAccuracy, top_k=1), ...}
        )
```

**What you may change.** Add task-shaped defaults to the same method — it is
the one place that answers "what does this data need?". Do not reintroduce a
live default in the constructor signature, and do not add a `multilabel:`
config knob: the target item type already says it, and a knob could disagree
with the data.

---

## 4. The TensorFlow backend is a subclass, and only of the loop

*2026-07-30*

**Context.** `TASKS.md` carried an open question for a year: does
`sonair tensorflow train` need a runnable at all? `KERAS_BACKEND=tensorflow
sonair keras train cfg.yaml` already executes the identical config on
TensorFlow, because Keras 3 IS `tf.keras` in TensorFlow >= 2.16. A backend whose
only content is "the Keras backend again" would make the project's central claim
— that a backend costs almost nothing — true by tautology rather than by
evidence.

**Decision.** Implement it, but only for what Keras 3 does not expose, and make
that boundary structural: `TensorFlowClassifier` **subclasses**
`KerasClassifier` and overrides sixteen names, every one of them about `tf.data`,
the traced loop, or the SavedModel.

- `_dataset` — `tf.data.Dataset.from_generator` over the same `RecordSequence`,
  plus `.prefetch(tf.data.AUTOTUNE)`.
- `_train_step_fn` / `_eval_step_fn` / `_predict_step_fn` — `tf.GradientTape`
  under `@tf.function`, which is what makes `jit_compile` (XLA) a knob.
- `_compile` — keeps the parent's NAME (the parent's `initialize()` calls it)
  and replaces the body: a compiled model owns its own training step, which is
  precisely what this backend exists to replace.
- `_export` — `keras.Model.export`, implemented only on the TensorFlow engine.

Everything else — the label peek, `LabelMap` fitting, `num_classes` resolution,
`_to_arrays`, the prediction contract, the artifacts — is *inherited*, not
copied.

**Consequences.**

- The tag is `framework="keras"`, not `"tensorflow"`. Record §2 settled that the
  axis names the API a class is wired TO, and this class is wired to
  `keras.losses`, `keras.metrics` and `KerasModel`. It joins the Keras picker
  alongside `KerasClassifier`, which is correct: both consume the same parts.
- There is therefore **no `tensorflow_classification.py`**. A second module would
  register the same `keras.losses` classes under a second framework tag, and a
  bare `!class:SparseCategoricalCrossentropy@role=loss` would become ambiguous
  everywhere navigaitor bootstraps every entry point — a real ergonomic
  regression bought with nothing.
- The constructor **restates every parameter of its parent** rather than taking
  `**kwargs`. Static introspection (`to_pydantic` → form specs, MCP schemas,
  editor widgets) reads the signature; `**kwargs` would type-check and erase the
  config surface.
- `_resolve_task_defaults` had to diverge, and the reason is exact: the Keras
  backend can default `metrics` to `["accuracy"]` because `compile()` resolves
  that alias *contextually*, against the loss and the target shape. A
  hand-written loop has no `compile()`, and `keras.metrics.get("accuracy")`
  returns the elementwise-equality function — which reports `0.0` forever
  against a raw-logits head. The defaults here are the concrete classes
  `compile()` would have chosen, and `_as_metric` maps the alias the same way.
- The override set is pinned by a test. Erosion here is the failure mode: a
  method that is really about *classification* drifting into this class would
  silently un-share it from every other backend.

**Example.**

```python
@tf.function(jit_compile=True if self.jit_compile else None)
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = network(x, training=True)
        loss = loss_fn(y, logits)
    variables = network.trainable_variables
    optimizer.apply_gradients(zip(tape.gradient(loss, variables), variables))
    for metric in metrics:
        metric.update_state(y, logits)
    return loss
```

**What you may change.** Add TensorFlow-specific behaviour (distribution
strategies, a different export format). Do not add classification behaviour —
that belongs on `KerasClassifier`, where every backend gets it. If a change here
would also be right for the Keras backend, it goes there instead.

---

## 5. A backend adds a module only when it adds a wiring surface

*2026-07-30*

**Context.** With one backend, "a backend is a runnable + a model wrapper + a
loss set" looked like the shape of the thing. Records §1 and §2 both describe
that shape, and `docs/frameworks.md` documented it as a three-step recipe. Then
two backends landed that need only the first: `TensorFlowClassifier` and
`FastaiClassifier`.

The temptation in both cases was symmetry — a `tensorflow_model.py` beside
`keras_model.py`, a `fastai_classification.py` full of re-registrations — so that
every backend would have the same three files. That symmetry would have been
pure duplication: `TensorFlowModel` would be `KerasModel` with the class name
changed, because `tf.keras.applications` *is* `keras.applications`.

**Decision.** A backend contributes a module only where it contributes something
a config can WIRE. Concretely:

| Backend | Runnable | Model wrapper | Loss/metric set |
|---|---|---|---|
| `lightning` | `lightning_classifier.py` | `timm_model.py` | `torch_classification.py` |
| `keras` | `keras_classifier.py` | `keras_model.py` | `keras_classification.py` |
| `tensorflow` | `tensorflow_classifier.py` | *reuses `KerasModel`* | *reuses the Keras set* |
| `fastai` | `fastai_classifier.py` | *reuses `TimmModel`* | `fastai_classification.py` |

fastai gets a loss module because its `*Flat` family is genuinely new wiring —
but tagged `framework="torch"`, since a Lightning run consumes those losses just
as well. It gets no model module because fastai *is* torch.

**Consequences.**

- `sonair/__init__.py`'s `_BACKEND_NAMES` gains one lazy name per backend, not
  three. `sonair[tensorflow]` and `sonair[fastai]` therefore advertise a smaller
  surface than the other two, which is honest: there is less to wire.
- The asymmetry is *visible in the docs* rather than smoothed over — the
  "what each layer owns" table in `docs/frameworks.md` says *sometimes* for two
  rows, and the config example shows the `fastai` scope block overriding two keys
  where `keras` overrides four.
- This is the strongest evidence the project has for its own claim. The `fastai`
  block in `examples/train_efficientvit_b0_mnist.yaml` changes the runnable class
  and the logger, and reuses the model, the preprocessing chain and the batch
  shape *verbatim*. Nothing had to be generalised to make that work — it fell out
  of the record model and the `@entrypoint` dispatch already being engine-free.

**Example.**

```yaml
# the whole cost of the fastai backend, in config
fastai: !scope:framework=fastai
  runnable: !class:sonair.FastaiClassifier
    metrics: [accuracy, top_k_accuracy]
  logger:
    - !class:fastai.callback.progress.CSVLogger
      fname: fastai_logs/mnist.csv
```

**What you may change.** Add a model wrapper or a loss module when the backend
genuinely introduces one. Do not add one for symmetry — a wrapper that differs
from an existing one only by name is a rename, and the config already says which
engine is running.

---

## 6. `KERAS_BACKEND` is pinned in three places, on purpose

*2026-07-30*

**Context.** Keras 3 binds its compute engine ONCE, at import time, from
`KERAS_BACKEND`. `recordstream.keras` sets a default — the first engine actually
*installed*, in the order torch → tensorflow → jax — which is right for a Keras
run (any engine will do) and wrong for `TensorFlowClassifier`, whose
`tf.GradientTape` loop cannot differentiate a torch-backed `keras.Model`. On the
workspace's own machine, which has `sonair[lightning]`, that default is `torch`.

The failure this produces is the bad kind: it surfaces deep inside the tape,
several minutes into a run, saying nothing about import order.

**Decision.** Three layers, each covering an entry path the previous one cannot
reach, plus a runtime check that turns any remaining miss into one clear line.

1. **`main()` → `_pin_keras_backend`** — the CLI path. It runs before liquifai
   resolves the config, which matters because resolving `!class:sonair.TensorFlowClassifier`
   *imports* the runnable, and therefore keras. A handler would be too late.
2. **`sonair/_tensorflow.py`** — the import path, for callers that never touch
   this CLI (`recordstream run cfg.yaml --scope framework=tensorflow`). It also
   re-exports the `keras` handle, so `tensorflow_classifier.py` imports keras,
   TensorFlow and the backend probe from ONE module — which removes an ordering
   hazard rather than documenting it: isort sorts `sonair._keras` *above*
   `sonair._tensorflow`, so importing the handle from its usual home would load
   keras one line before the default meant to steer it.
3. **`TensorFlowClassifier.initialize()`** — the check. If keras is already bound
   to another engine, it raises with the engine it found, the variable to set,
   and the alternative (`sonair keras`, which runs the same config on whatever is
   loaded).

All three use `setdefault`, so an explicit `KERAS_BACKEND=jax` still wins and is
then *reported* rather than silently overridden.

**Consequences.**

- The import ordering in `tensorflow_classifier.py` is load-bearing, so it is
  pinned by tests that read the module as TEXT — and those tests deliberately run
  even where TensorFlow is absent, because the machines that can break the
  ordering are exactly the ones without it.
- **One engine per process** is inherited from Keras and cannot be worked around.
  A full test run imports `test_keras_classifier.py` first and binds torch, so
  the TensorFlow loop tests skip and are run as their own invocation
  (`KERAS_BACKEND=tensorflow pytest tests/test_tensorflow_classifier.py`). A
  marker that pretended otherwise would report coverage that never executed.

**Example.**

```python
# sonair/_tensorflow.py — the setdefault MUST precede both imports
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import tensorflow as tf                                    # noqa: E402
from recordstream.keras import keras, keras_backend        # noqa: E402
```

**What you may change.** Add a fourth entry path's pin if one appears. Do not
replace `setdefault` with an assignment (it would silently override a deliberate
choice), and do not drop the `initialize()` check because "the CLI already sets
it" — the check is what makes the other two debuggable when they miss.
