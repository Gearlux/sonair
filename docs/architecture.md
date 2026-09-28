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

*(Superseded by §1b: the sub-apps are gone and the tables are one table. What
survives is the shape of the answer — data plus a factory, no registry module.)*

### 1b. …and the eighth backend removed the sub-apps entirely

*2026-08-01*

**Context.** With eight backends the CLI mounted 24 commands, and a question
worth asking late is what any of them actually *did*. The answer, for every one:

```python
def command(runnable: Any) -> None:
    if spec.registrations is not None:
        _register_backend_classification(spec.registrations)
    _dispatch(runnable, VERB_TASKS[verb])
```

…plus one line in `main()`:

```python
argv += ["--scope", f"framework={backend}"]     # _inject_framework_scope
```

The backend verb never *selected* a backend. It activated a scope, and the
config's `!scope:framework=<name>` block did the selecting. Two things followed,
and both were measured rather than reasoned about:

1. **Nothing compared the verb to the config.** `sonair keras train <a config
   with no keras block>` resolved to `LightningClassifier`, imported the Keras
   loss set, and trained — silently. So did any typo, since an unmatched scope
   value fell through to the document's unscoped keys.
2. **The sub-apps differed in two values.** A name and a registrations module.
   §1a had already collapsed twelve handlers into a table for the same reason;
   the sub-apps themselves were the part that had not been collapsed.

Meanwhile liquifai binds `--KEY VAL` for any dimension a config declares
(`flags.bind_dimension_flags` → `confluid.discover_dimensions`), so
`recordstream run cfg.yaml --framework keras` already worked, with no sonair code
involved at all. The subcommand layer existed to spare the user from typing a
flag that the framework underneath had always provided.

**Decision.** Three verbs, one flag, one table.

```bash
sonair train cfg.yaml --framework keras     # was: sonair keras train cfg.yaml
sonair frameworks                           # was: what `sonair --help` listed
```

- `FRAMEWORKS: Dict[str, Framework]` — name → description + registrations module.
  The verbs are built once, for all backends, so a new row adds no commands.
- `--framework` is **rewritten** to `--scope framework=<name>` in `main()`, not
  appended. liquifai binds the ergonomic flag only for *declared* dimensions, so
  against a config with no `framework` blocks a bare `--framework keras` would be
  read as a config override; rewriting makes it mean one thing either way.
- The check the verb could never do now lives one layer down: confluid 0.3.0
  raises when an active scope value matches no block, listing the values the
  document declares. A backend named at the command line is verified against the
  config that has to implement it.

**Consequences.**

- `TEMPLATES`, `_backend_app`, `_template_app`, `_template_message`,
  `_template_command`, `BackendNotImplementedError` and
  `_inject_framework_scope` are all **deleted**. An unbuilt backend is absent
  from `FRAMEWORKS`; there is nowhere for a stub to hang on a flag, so the
  roadmap-advertising machinery went with the sub-apps. `sonair frameworks`
  lists what is real, which is a smaller and truer claim than a menu of half.
- **A backend with no `--framework` registers no losses.** The flag is what names
  the engine, and there is no safe guess: importing the default's set would drag
  torch into a keras-only install. The failure mode is a loud `UnknownClassError`
  on a bare registered name, not a wrong result.
- **The default backend is a block like every other, named by `default_scopes:`.**
  `--framework lightning` must name something, and a bare run must pick
  something, so the example carries `default_scopes: [framework=lightning]` and a
  `lightning: !scope:framework=lightning` block holding the runnable (2026-08-18;
  before confluid grew `default_scopes:` the top level *was* the default and the
  block restated it). The confluid check binds the default too — a typo'd one is
  the same error as a typo'd flag — so a config states which backends it
  supports, and which one it means, instead of leaving either to be inferred.
- **A breaking CLI change**, paid once — 18 files and ~61 command spellings, in
  the same style as the `FlaxModel` → `FlaxConvNet` rename.

**Example.**

```python
FRAMEWORKS: Dict[str, Framework] = {
    "keras": Framework("Keras 3 — one API over PyTorch / TensorFlow / JAX.",
                       "sonair.keras_classification"),
    "jax":   Framework("JAX — a jitted optax loop over a Flax model."),  # nothing to register
}

def main() -> None:
    framework = _requested_framework(sys.argv)   # argv: both steps below precede config load
    _pin_keras_backend(framework)
    _register_classification(framework)
    _normalize_framework_flag(sys.argv)          # --framework X -> --scope framework=X
    app.run()
```

**What you may change.** Add rows to `FRAMEWORKS`. Do not reintroduce a backend
subcommand, and do not grow `Framework` to carry per-backend *behaviour* —
`_pin_keras_backend` would be a third column that is `None` seven times out of
eight, which is why it stays visible in `main()`.

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
  (inferring the engine from a class's base), mirroring the long-standing consumer-side
  inference of a class's role.
- **Exclusion requires two claims.** A candidate that declares no framework is
  never filtered out, and a runnable that declares none filters nothing. Every
  untagged project (raidar, deltaid, waivefront) therefore behaves exactly as
  before — the axis is opt-in per class, not a flag day.
- **Framework is enforcement, task is guidance.** A discovery service deliberately offers
  cross-*task* options (a segmentation model in a classification model slot — "a
  model is a model"), ranked after the preferred tier. Cross-*framework* options
  are not offered at all, because they are not a choice: they cannot run.
- **A visual editor followed on 2026-07-31**, once it had its own decision — the axis
  reached the canvas without the workflow-format break this record feared. Its
  object sockets now carry the engine as a third segment inside a comma UNION
  (`RECORDSTREAM_OBJECT:loss:torch,RECORDSTREAM_OBJECT:loss`), which both
  ComfyUI halves read as "compatible if the sets intersect" — so a saved graph
  keeps loading and an untagged class keeps connecting. The union carries the
  claim; the *rule* is applied by a backend `VALIDATE_INPUTS` and a canvas
  `onConnectInput` hook, because no arrangement of type strings can express
  "exclusion requires two claims" on its own.
- **One registration module per engine** — `torch_classification.py` (2026-07-30,
  renamed from `classification.py`, which read as task-scoped in a project whose
  whole point is that the task outlives the engine) and `keras_classification.py`.
  The tag alone would not have been enough: each module is its own
  `confluid.configurables` entry point, so on a torch-only install the Keras
  module fails to import and the scanner skips it — the registrations are absent
  because the engine is, which is the honest answer rather than a set of names
  that cannot be built. Splitting also keeps the import cost paid only by whoever
  installed that backend.
- **The entry-point list is DERIVED from the tree by a test** (2026-08-01). One
  module per registration is a good decision with a bad failure mode: forgetting
  a line costs nothing a developer would notice. A dotted path in YAML imports
  the module itself and the CLI imports its own backend, so configs and runs keep
  working — only the discovery surfaces go blind, and they fail by *offering
  fewer options*, which looks like a short list rather than a bug. Five backends
  duly forgot: the huggingface / jax / mlx / tensorflow-model work of 2026-07-31
  shipped eight registering modules with no entry point. `tests/
  test_entry_points.py` now finds the expectation by `ast`-scanning the package
  for `@configurable` classes and `configurable(...)` / `register(...)` calls,
  and asserts each such module is declared. Static on purpose — no venv has every
  backend installed, so an import-based scan would reduce the check to whatever
  the local extras happen to be.
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
            {"accuracy": PartialClass[Metric](MultilabelAccuracy), ...}
            if self.multilabel
            # `accuracy`, the word EVERY engine can produce — see record 11.
            else {"accuracy": PartialClass[Metric](MulticlassAccuracy, top_k=1), ...}
        )
```

**What you may change.** Add task-shaped defaults to the same method — it is
the one place that answers "what does this data need?". Do not reintroduce a
live default in the constructor signature, and do not add a `multilabel:`
config knob: the target item type already says it, and a knob could disagree
with the data.

---

## 4. The TensorFlow backend is a subclass, and only of the loop

*2026-07-30 — **SUPERSEDED by §9** (2026-08-01): the subclass relationship was
retired and `TensorFlowClassifier` now extends `ProgressReporting` directly. The
record stands as written because its Context and its list of what the backend
adds are still the reasons the backend exists; only the "subclasses
`KerasClassifier`" mechanism is out of date. §9 records why.*

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
  everywhere a discovery service bootstraps every entry point — a real ergonomic
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
| `keras` | `keras_classifier.py` | `keras_model.py` ‡ | `keras_classification.py` |
| `tensorflow` | `tensorflow_classifier.py` | `tensorflow_model.py` † ‡ | *reuses the Keras set* |
| `fastai` | `fastai_classifier.py` | *reuses `TimmModel`* | `fastai_classification.py` |
| `huggingface` | `huggingface_classifier.py` | `huggingface_model.py` | *none — the model computes its own loss* |
| `jax` | `jax_classifier.py` | `convnet/flax_convnet.py` ‡ | *none — optax ships functions, not classes* |
| `mlx` | `mlx_classifier.py` | `convnet/mlx_convnet.py` ‡ | *none — `mlx.nn` ships classes, but as names not tags* |

fastai gets a loss module because its `*Flat` family is genuinely new wiring —
but tagged `framework="torch"`, since a Lightning run consumes those losses just
as well. It gets no model module because fastai *is* torch.

### † The amendment: `tensorflow` DOES contribute a model module (2026-07-31)

The reasoning above was right about what it examined and wrong about its scope.
It said a `TensorFlowModel` would be `KerasModel` with the name changed, because
`tf.keras.applications` *is* `keras.applications` — true, and it rules out a
duplicate **zoo wrapper over keras.applications**. It does not rule out anything
else, and `keras.applications` is not TensorFlow's only model source. Two things
landed that duplicate nothing:

- **`TensorFlowModel`** loads a **SavedModel** (`tf.saved_model.load`) — the
  format Kaggle Models serves, where TF Hub redirected, and what any
  `model.export()` writes. A different zoo, no new dependency. (`tensorflow-hub`
  is deliberately unused: last release 2024-01-30.)
- **`TensorFlowConvNet`** implements the shared conv spec in raw `tf.nn` ops on a
  `tf.Module`, wrapping nothing at all.

Both are tagged `framework="keras"`, which reads wrong and is deliberate: the
axis names the API a class is **wired to** (§2), and `TensorFlowClassifier` — the
only trainer that can consume them — declares `"keras"` itself. A `"tensorflow"`
tag would describe the imports better and hide both classes from the one slot
picker that should offer them.

Two consequences the implementation had to handle, both measured rather than
anticipated:

- A `tf.Module` has no `keras.Model.export`, so `_export` branches on the
  capability and falls back to `tf.saved_model.save` with an explicitly traced
  signature.
- `tf.saved_model.load` restores tracked DATA, not the Python class, so
  `tf.Module.variables` — a computed property — is **gone** from the restored
  object. A Keras export reports its variables; a module export reports none
  while its children hold all of them. `_restored_trainable_variables` therefore
  takes the declared list when there is one and otherwise walks the trackable
  tree. Without that, every SavedModel this project exports would look
  un-fine-tunable.

**The rule this generalises.** A decision record's rationale is scoped to what it
examined. When a new case arrives, check whether the *reasoning* covers it before
citing the *conclusion* — "no `TensorFlowModel`" was a conclusion about zoo
wrappers, and reading it as a ban on the module cost a plan revision.

### ‡ The second amendment: the convnets are a package, apart from the zoo wrappers (2026-08-01)

The amendment above landed `TensorFlowModel` and `TensorFlowConvNet` in one
file, and `KerasConvNet` had landed in `keras_model.py` beside `KerasModel` for
the same reason: both are "the Keras model module". That grouping is by
*backend*, and the more useful axis is what the class **wraps**.

A zoo wrapper exists to reach somebody else's weights. Its knobs are about the
zoo — a Hub id, `pretrained`, a SavedModel path — and its value is that it is
thin. A convnet implements [`convnet/spec.py`](../sonair/convnet/spec.py) and
wraps nothing; its knobs are the architecture (`widths`, `dropout`, `seed`), and
its reason to exist is a claim about the **other** convnets, not about its own
backend: that every framework implements the same network. The two changed for
different reasons and were read by different people.

Splitting them into `<backend>_model.py` / `<backend>_convnet.py` fixed that and
immediately exposed a second problem it could not fix. The implementations (four
at the time, five once `TorchConvNet` landed) are peers in fact but were not peers
in the file listing: `keras_convnet.py` and
`tensorflow_convnet.py` sat next to their zoo wrappers, while the JAX and MLX
ones — `flax_convnet.py`, `mlx_convnet.py` — looked like zoo wrappers themselves,
because those backends have no zoo wrapper to be distinguished from. Nothing
about a flat directory says these four are the same network.

**Decision.** A directory does what a naming convention could not. Every
implementation of the spec lives in the **`sonair.convnet` package** and the zoo
wrappers stay at the top level, one `<backend>_model.py` each. Membership of the
package is what says "this is a spec implementation". Each convnet keeps its own
`confluid.configurables` entry point and its own test file.

**The filename still names the class.** Package membership carries the
*grouping*, so the modules were briefly renamed to their framework alone
(`keras.py`, `tensorflow.py`, `flax.py`, `mlx.py`) on the reasoning that the
filename need not repeat what the directory says. That was wrong, and the cost
was concrete: every other module in this project is named for the class it
exports — `timm_model.py` → `TimmModel`, `keras_model.py` → `KerasModel`,
`lightning_classifier.py` → `LightningClassifier` — so the package became the one
place where `ls` and `grep KerasConvNet` disagreed about where a class lives, in
exchange for symmetry the directory was already providing. The modules are
therefore `keras_convnet.py`, `tensorflow_convnet.py`, `flax_convnet.py` and
`mlx_convnet.py`, each the snake_case of its exported class. `spec.py` keeps its
name because it exports no class — only the constants and
`trainable_parameter_count`.

**The classes were renamed too, and that is the part the directory could not
do.** The JAX and MLX implementations were called `FlaxModel` / `MlxModel` in
`flax_model.py` / `mlx_model.py`, because when they landed there was no zoo
wrapper beside them for the name to distinguish them from. Inside the package
that reasoning is gone: the neighbours are `KerasConvNet` and
`TensorFlowConvNet`, and "Model" now says the one thing that is false about
them — that they wrap somebody's weights. So they are `FlaxConvNet` and
`MlxConvNet` in `flax_convnet.py` / `mlx_convnet.py`, and the package is
uniform: four `*ConvNet` classes in four `*_convnet.py` modules. Renaming a
public class is a real cost (a YAML wires it: `!lazy:sonair.FlaxConvNet`), paid
once, here, rather than left as a permanent footnote.

**Each convnet IS its framework's model type.** `TorchConvNet` is a
`torch.nn.Module`, `KerasConvNet` a `keras.Model`, `TensorFlowConvNet` a
`tf.Module`, `FlaxConvNet` an `nnx.Module`, `MlxConvNet` an `mlx.nn.Module` —
the same decision as `TimmModel` being a `torch.nn.Module` (§7). `MlxConvNet` was the last holdout, a plain class
whose network the MLX loop reached as `self.model.model`; that made
`MlxClassifier.model` typed as the concrete wrapper, the only backend whose slot
could not hold a network the user brought. It is `Partial[nn.Module]` now.

Two MLX specifics that shape the subclass, both measured:

- `nn.Module.valid_parameter_filter` ends in `and not key.startswith("_")`, so
  the lazily-built network lives in the **public** attribute `net`. As `_model` —
  the name its Keras and TensorFlow twins use — it is filtered straight out of
  `parameters()`, the optimizer sees nothing, and the loop applies gradients to
  an empty tree forever without erroring.
- MLX carries train/eval as module **state**, not as a per-call argument, so
  `__call__(x, training=None)` leaves the mode alone by default. A `training:
  bool = False` default (what the Keras and Flax twins take) would flip the
  network to eval on every forward — including the one `nn.value_and_grad` makes
  inside the training step, which passes no flag. The classifier therefore sets
  the mode once per phase (`self.model.train(False)` before an eval pass) rather
  than per call, which is also what keeps the widened slot honest: a bare
  `nn.Module` has no `training=` keyword to accept.

**`sonair/convnet/__init__.py` imports nothing**, which is load-bearing rather
than stylistic: `sonair/__init__.py` resolves every backend-owned name lazily so
a Keras-only install never imports torch, and an `__init__` re-exporting the four
classes would import keras *and* tensorflow *and* flax *and* mlx the moment
anything under the package — including the dependency-free `spec.py` — is
touched.

**Consequences.** `keras_model.py` no longer imports the spec at all, and
`tensorflow_model.py` shed the constants plus the private `_ConvBlock` /
`_ConvNet` — roughly 150 lines of hand-written batch-norm that a reader looking
for "how do I load a SavedModel?" used to walk past. `ls sonair/convnet/` is now
the answer to "what implements the shared spec?", where the answer used to be a
grep across the whole package. The cost is one more level in every import path.

**Example.** The move itself needed no config change — the lazy names in
`sonair/__init__.py` absorb it. The class renames did, and they are the only
breaking part: a config saying `!lazy:sonair.FlaxModel` must now say
`!lazy:sonair.FlaxConvNet`.

```yaml
model: !lazy:sonair.KerasModel      # keras_model.py            — a keras.applications backbone
model: !lazy:sonair.KerasConvNet    # convnet/keras_convnet.py  — the shared spec, wrapping nothing
model: !lazy:sonair.FlaxConvNet     # convnet/flax_convnet.py   — the same spec, in flax.nnx
model: !lazy:sonair.MlxConvNet      # convnet/mlx_convnet.py    — …and in mlx.nn
```

### ‡ The torch entry, and the one backend that gets no convnet (2026-08-01)

`TorchConvNet` completed the set, and it is worth recording *why it was last*
rather than first: torch was the only backend whose zoo wrapper (`TimmModel`,
~1000 pretrained backbones) is so complete that nothing forced the question. The
spec exists because Flax and MLX have no zoo at all; torch has the best one. So
the torch implementation buys nothing for torch *users* — it buys the
**comparison**, by making the from-scratch baseline available on the framework
everything else is measured against.

It is the one **channels-FIRST** implementation. The other four consume NHWC
because keras, tf, flax and mlx all default to it; torch is NCHW. The
constructor knob deliberately does not follow the tensor — `input_shape` stays
`(H, W, C)` in all five, so one config line drives any of them, and only `C`
reaches the parameter count. Reading it as NCHW instead would build
`Conv2d(224, …)` rather than `Conv2d(3, …)`: still runnable, silently a
different network, which is why a test pins `in_channels == 3`.

**Five convnets for eight backends, and neither shortfall is a gap.** Three
backends run on torch — `lightning`, `torch` and `fastai` — and all three take
`TorchConvNet` unchanged, the same reason they already share `TimmModel`; there
is no `fastai_convnet.py` for the same reason there is no `fastai_model.py` (§5:
a backend adds a module only where it adds wiring surface). `tensorflow` can take
`KerasConvNet` too, since `tf.keras` IS Keras 3. The genuine absence is **`huggingface`**: `transformers.Trainer` calls
the model with `labels=` and subscripts `outputs["loss"]`, so it drives models
that compute their own loss from `config.problem_type`. A convnet returning bare
logits does not satisfy that contract, and satisfying it would mean a
`PreTrainedModel` subclass carrying a `transformers` config — wrapping the
convnet in exactly the zoo machinery the spec exists to avoid. That is a
deliberate non-goal, not a backlog item.

**What you may change.** A backend that brings both kinds brings both: a
`<backend>_model.py` at the top level and a `<backend>_convnet.py` in the
`convnet/` package. A backend with no zoo to wrap brings only the second — that
is `jax` and `mlx`. Do not flatten the package back to the top level, do not
rename a module away from the class it exports, and do not add a convnet that is
not an instance of its framework's own module type.

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
  and its metric names, and reuses the model, the preprocessing chain, the batch
  shape and the trackers *verbatim*. Nothing had to be generalised to make that
  work — it fell out of the record model and the `@entrypoint` dispatch already
  being engine-free.

**Example.**

```yaml
# the whole cost of the fastai backend, in config
fastai: !scope:framework=fastai
  runnable: !class:sonair.FastaiClassifier
    metrics: [accuracy, top_k_accuracy]   # NAMES: fastai metrics are plain functions
  val_metrics: null                       # the torchmetrics above have no reader here
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

## 7. A model wrapper subclasses its engine's model base

*2026-07-31*

**Context.** `KerasModel` wraps a `keras.applications` backbone plus a fresh
head. It began as a plain class holding the built network in `self._model` and
exposing it as a `model` property — so every call site that wanted to *drive*
the network wrote:

```python
network = getattr(self.model, "model", self.model)
network.compile(...)
```

That line appeared three times across two backends, and it is doing something
subtle: a bare `keras.Model` has **no `.model` attribute** (measured), so the
`getattr` fallback is spanning two incompatible shapes at every use. The
asymmetry was the tell — the torch twin needs no such line, because `TimmModel`
IS an `nn.Module` and a `LightningModule` can simply call it.

Delegating the surface by hand (`def compile(self, **kw): self.model.compile(**kw)`,
and the same for `fit` / `evaluate` / `predict` / `export` / `save`) was tried
first. It works, but it re-declares in the wrapper an interface that inheritance
gives for free, and every member added upstream is one the wrapper silently
lacks.

**Decision.** The wrapper subclasses its engine's model base, mirroring the torch
side exactly: `TimmModel(nn.Module)` ⇒ `KerasModel(keras.Model)`. `compile`,
`fit`, `evaluate`, `predict`, `export`, `save`, `save_weights` and
`trainable_variables` are then inherited, and a classifier addresses whatever is
wired into its `model` slot with no unwrapping step — the wrapper and a bare
`keras.Model` are the same kind of thing.

Two members remain hand-written, each for a measured reason:

- **`build(input_shape)`** carries the lazy build. Keras forbids a layer from
  adding sub-layers once built, so creating the network on the first `call()`
  dies with *"You cannot add new elements of state … to a layer that is already
  built"*. `build()` runs before that first call, which is what keeps a
  directly-constructed, never-flowed instance usable. The normal path never
  reaches it — `flow()` fires `solidify()` first, and the cache makes the second
  trigger free.
- **`output_shape`** is Functional-only; a subclassed model raises
  `AttributeError` for it. The inner network IS functional, so it is one hop.

**Consequences.**

- The zero-arg / lazy-init mandate still holds: `KerasModel()` constructs, builds
  nothing, and validates required config at build time.
- `_compile()` now compiles the WIRED OBJECT, so its Keras-owned state
  (`loss`, `_compile_metrics`) reads back off `self.model` directly rather than
  off an inner network.
- `trained_model` returns the wrapper. That is a fix, not a side effect: the
  wrapper carries `num_classes`, which an evaluation run reads straight off its
  model slot, so the output now docks into a downstream classifier — a bare
  `keras.Model` never could.
- The `getattr` had been doubling as a TYPE escape hatch (it returns `Any`), so
  deleting it surfaced 14 mypy `union-attr` errors: `self.model: Partial[Any]` is
  `Union[Any, Fluid]`, and a method call through the `Fluid` arm is rejected. The
  answer is not an accessor — one was written and thrown away — but the
  annotation: the ATTRIBUTE carries the real type (`self.model: keras.Model`)
  while the CONSTRUCTOR param keeps the `Partial[...]` marker, which is the only
  place it is machine-read. Restating `Partial` on the attribute bought nothing and
  cost every direct call.
- **The param's `T` was narrowed on 2026-08-01** — it read `Mandatory[Partial[Any]]`
  here and in five sibling backends, which declared the deferral and nothing
  else. Each now names what it takes (`Partial[keras.Model]` here), and the two
  slots that genuinely admit two types name a union alias rather than falling
  back to `Any`: `TensorFlowNetwork = Union[keras.Model, tf.Module]` and
  `HuggingFaceNetwork = Union[HuggingFaceModel, torch.nn.Module]`. `Partial[T]`'s
  own docstring blesses `Partial[Any]` "when the target type is genuinely open" —
  none of these was. See §8.

**Example.**

```python
@configurable(task="classification", role="model", lazy=True, framework="keras")
class KerasModel(keras.Model):
    def build(self, input_shape=None):   # the ONE place a subclass may create state
        self.solidify()
        super().build(input_shape)

    def call(self, x, training=False):   # the twin of TimmModel.forward
        return self.model(x, training=training)

# ...so the classifier just drives it, and a bare keras.Model works identically:
self.model.compile(optimizer=..., loss=..., metrics=[...])
```

**What you may change.** Add a member only when inheritance genuinely cannot
supply it, and say which measured behaviour forced it — the two above name
theirs. Do not reintroduce `getattr(self.model, "model", self.model)`: if a call
site seems to need it, the wrapper is missing a base class, not a fallback.

---

## 8. A deferred slot names its type; `Partial[Any]` is half a declaration

*2026-08-01*

**Context.** Every backend's `model` is a deferred slot: a config wires
`model: !lazy:sonair.TimmModel` and the classifier flows it at run time with the
dataset-derived width (`flow(self.model, num_classes=…)`). `Partial[T]` —
`Annotated[Union[T, Fluid], "__confluid_partial__"]` — is what declares that: the
marker is machine-read by `confluid.partial_param_names`, so liquifai's auto
deep-flow skips the slot instead of building a target that still needs a runtime
kwarg, and the serializer emits `!lazy:` rather than `!class:`.

Six of the seven backends that existed then spelled it `Mandatory[Partial[Any]]`.
That declares the deferral and nothing else, and three arguments had accumulated
for leaving it there: the ATTRIBUTE already carries the real type (§7), a
narrower `T` buys nothing in the generated schema for a torch model, and some
slots really do admit two unrelated types. Only the first is a reason to keep
`Any` on the attribute; none is a reason to keep it on the param.

**Decision.** Each backend's `model` param names what that backend accepts:

| backend | slot type | why |
| --- | --- | --- |
| lightning, torch, fastai | `Partial[torch.nn.Module]` | all three run on torch; fastai's `initialize()` already raised `TypeError` for a non-module |
| keras | `Partial[keras.Model]` | the class calls `compile`/`fit`/`export` through the slot |
| tensorflow | `Partial[TensorFlowNetwork]` | `Union[keras.Model, tf.Module]` — see below |
| jax | `Partial[nnx.Module]` | `nnx.Optimizer(model, …)` and `nnx.value_and_grad` require one |
| mlx | `Partial[nn.Module]` (`mlx.nn`) | `nn.value_and_grad(model, …)` / `optimizer.update(model, …)` take the module itself |
| huggingface | `Partial[HuggingFaceNetwork]` | `Union[HuggingFaceModel, torch.nn.Module]` — `_network()` unwraps either |

Six of the eight name their framework's own module BASE rather than a concrete
sonair class, and that is possible because of a second decision made the same day:
every implementation of the shared conv spec IS its framework's module type —
`TorchConvNet(nn.Module)`, `KerasConvNet(keras.Model)`,
`TensorFlowConvNet(tf.Module)`, `FlaxConvNet(nnx.Module)`,
`MlxConvNet(mlx.nn.Module)` — the same rule §7 sets for the zoo wrappers. While
`MlxConvNet` was still a builder holding its network in an attribute, the MLX slot
could only be typed as that one class; once it became an `nn.Module`, the slot
became `nn.Module` and a user's own module fits it.

Where two types genuinely meet in one slot, the union is spelled out as a
module-level alias with a `#:` comment, never widened back to `Any`.
`TensorFlowNetwork` is the real case: under the TensorFlow backend a
`keras.Model` descends `Layer -> TFLayer -> KerasAutoTrackable`
(`keras/src/backend/tensorflow/trackable.py`), so it is NOT a `tf.Module` and the
nearest shared ancestor is `tf.__internal__.tracking.AutoTrackable` — a private
API this must not name. `HuggingFaceNetwork`'s second arm is `torch.nn.Module`
rather than `transformers.PreTrainedModel` because every `AutoModelFor*` is one,
and naming the transformers class would force a module-level `import
transformers` that this backend defers to the methods that need it.

**Consequences.**

- What the narrowing buys differs by arm, and both halves were measured. For a
  torch model it changes NOTHING downstream: `confluid.pydantic_export`
  coerces `torch`/`numpy` types to `Any` (`_OPAQUE_TOP_MODULES`), so
  `Partial[nn.Module]` and `Partial[Any]` generate the identical field. For the rest it
  is real — `to_pydantic(HuggingFaceClassifier)` now offers the wrapper's own
  `HuggingFaceModelConfig` in the `model` field where it used to offer an open
  `{}`, because `HuggingFaceModel` is `@configurable` and the exporter recurses
  into it.
- The generated JSON schema for a narrowed slot loses the "anything goes" arm:
  `KerasClassifier.model` reports `$ref: FluidConfig` where it used to report
  `anyOf: [{}, FluidConfig]`. Pydantic still VALIDATES a live instance
  (`arbitrary_types_allowed`, verified for both a `keras.Sequential` and a
  `PartialClass` marker); it is the JSON projection that drops the un-schemable arm.
  A form-spec consumer therefore sees "wire a config here", which is what the
  slot is for.
- `TensorFlowClassifier` is the only slot where two FRAMEWORKS meet (the
  huggingface union is one framework's wrapper beside its own model), and it tells
  the arms apart where they differ rather than at the boundary — `_export`
  branches on `hasattr(model, "export")`, `_restored_trainable_variables` on the
  declared variable list.
- The narrowing has RUNTIME teeth, not just static ones: `@configurable` validates
  constructor kwargs against `to_pydantic(cls)` in strict mode
  (`confluid.decorators` → `validate_kwargs`), so a live object that is not an
  instance of the declared type is now rejected AT CONSTRUCTION instead of
  failing later inside the loop. The normal wiring path is unaffected — a
  `!lazy:`/`!class:` marker is a `Fluid`, and `validate_kwargs` deliberately skips
  deferred fields ("validate post-flow"). What it caught immediately was tests
  passing `model=None`.
- Tests that never touch the model now wire a cheap legal stand-in instead of
  `model=None`: a `_NO_MODEL = torch.nn.Identity()` constant in the torch-side
  suites, and a `_no_model()` FACTORY returning `tf.Module()` in the TensorFlow
  one — a factory there because that file's source-only tests must keep running
  where TensorFlow is absent. `model` is `Mandatory`; None was never legal, and
  28 call sites had been passing it under cover of `Any`.
- The root `mypy.ini` gained `[mypy-jax.*]` / `[mypy-flax.*]` / `[mypy-optax.*]`
  ignore sections, matching the ones fastai and mlx already had: they back a
  selectable extra, so they are absent from a default install and their imports
  would otherwise read as real missing imports.

**Example.**

```python
# tensorflow_classifier.py — the one slot where two frameworks meet
TensorFlowNetwork = Union[keras.Model, tf.Module]

class TensorFlowClassifier(ProgressReporting):
    def __init__(self, model: Mandatory[Partial[TensorFlowNetwork]], ...) -> None:
        # PARAM declares the deferral (machine-read by `partial_param_names`);
        # ATTRIBUTE declares what it holds once flowed — see §7.
        self.model: TensorFlowNetwork = model

# ...and everywhere else the type is simply the framework's own base:
class TorchClassifier(TorchRunner, ProgressReporting):
    def __init__(self, model: Mandatory[Partial[nn.Module]], ...) -> None:
        self.model: nn.Module = cast(nn.Module, model)
```

**What you may change.** A new backend picks the type its loop actually requires
and says so; if it needs two, add a `<Backend>Network` union alias beside the
other `#:`-documented ones. `TorchClassifier` was the first backend written after
this record and it landed at `Partial[nn.Module]` with no prompting, which is the
evidence the rule costs nothing to follow. `Partial[Any]` stays legal in confluid for
a genuinely open slot — no sonair `model` slot is one, and
`tests/test_lazy_slots.py::test_no_classifier_declares_its_model_slot_as_lazy_any`
reads the SOURCE (most backends need a framework a default install lacks, so an
import-based check would skip exactly where a copy-pasted slot lands).

---

## 9. A backend is a sibling, not a subclass (2026-08-01)

**Context.** `TensorFlowClassifier` subclassed `KerasClassifier` from the day it
landed, and record 4 justified that: in TensorFlow >= 2.16 `tf.keras` **is** Keras 3,
so the model wrapper, the losses, the metrics and the batch shape really are the same
objects. The inheritance was not decorative — fourteen of the parent's twenty-one
methods came through untouched, 163 of its 307 lines, and the TensorFlow `fit()` and
`predict()` genuinely called them.

Two facts about the code argued the other way, and neither is a matter of taste:

- Of the seven methods the subclass overrode, **five never called `super()`**. `fit`,
  `test`, `predict`, `_compile` and `_resolve_task_defaults` replaced the parent's
  bodies outright. A base class whose behaviour is entirely replaced is carrying code
  reuse, not substitutability.
- It **narrowed a precondition**. `_require_tensorflow_backend()` raises where a
  `KerasClassifier` runs fine, so a picker offering "any `KerasClassifier`" can hand
  out one that cannot run — and the framework axis cannot catch it, because both
  declare `framework="keras"`.

**Decision.** The two are siblings. `TensorFlowClassifier` extends
`ProgressReporting` directly and owns a full copy of the classification half.

The obvious alternative — extract an engine-neutral base and have both inherit it —
is **deferred, not rejected**. All eight backends carry that sequence now — every
`<backend>_classifier.py` has its own `_initialize_fit` — and the detection projects
have not been through it yet; a base extracted today would be fitted to classification
eight times and then reshaped for a detector's targets, boxes and metrics. It waits
for those consumers (`TASKS.md`).

**Consequences.**

- The label pipeline, the batch shape, the prediction contract and the artifact
  writers are duplicated. That is the price, and it is paid knowingly.
- The drift guard is **behavioural, not structural**: both classes are driven over
  the same string-labelled records and must agree on `num_classes`, `multilabel`,
  `class_names`, the batch shape and the target dtype. Comparing SOURCE would fail on
  docstrings and error text, which legitimately differ.
- Importing `sonair.tensorflow_classifier` no longer registers `KerasClassifier` as a
  side effect. A discovery service was never relying on that (each module is its own
  entry point), but `tests/test_categories.py` was.
- `RecordSequence` had to be re-exported from `sonair/_tensorflow.py`. A standalone
  module importing it from `recordstream.keras` would let isort sort that line into
  the third-party group above every `sonair` line — and `recordstream.keras` is where
  the competing `KERAS_BACKEND` default lives. Measured: with the variable unset,
  `import recordstream.keras` leaves it at `torch`.

**Example.**

```python
class TensorFlowClassifier(ProgressReporting):        # was: (KerasClassifier)
    def initialize(self) -> None:
        self._require_tensorflow_backend()            # the narrowed precondition
        self._initialize_fit() if self.task == "fit" else self._initialize_eval()
        self._compile()
```

**What you may change.** Extract the shared base once raidar and deltaid are done —
and do not restore the subclass relationship on the way. A backend that genuinely
specializes another (calling `super()`, widening nothing, narrowing nothing) may still
subclass it; none currently does.

---

## 10. A raw-TensorFlow module may not touch `tf.initializers` (2026-08-03)

**Context.** `TensorFlowConvNet` is the shared conv spec written in raw `tf.nn` ops — that is its
whole reason to exist beside `KerasConvNet`, which is the *Keras* entry and runs on whichever
engine Keras is bound to. It built its weights with `tf.initializers.GlorotUniform`, which looks
like TensorFlow and is not.

**Decision.** Compute Glorot-uniform weights from `tf.random.stateless_uniform` locally, and treat
every `tf.keras` / `tf.initializers` / `tf.losses` / `tf.metrics` / `tf.optimizers` reach-through
as a Keras import in disguise.

**Consequences.** Since TF 2.16, `tf.keras` **is** Keras 3, so `tf.initializers.GlorotUniform` is
`keras.initializers.GlorotUniform` — `is`-identical, measured, not a wrapper. In a process where
Keras is already bound to the **torch** engine (this workspace's default, since
`recordstream.keras` picks the first engine installed and torch wins) it returns a **torch
tensor**, and `tf.Variable` then dies converting an MPS tensor to numpy.

The failure is invisible until both engines share one interpreter: the same file passes alone and
fails after any Keras test has run. That is why it was found in the *segmentation* project's twin
of this module, whose suite exercises both in one session — and why the guard here is a SOURCE
scan across the package that runs even on a machine with no TensorFlow installed, since those are
exactly the machines where the next such module gets written.

`stateless_uniform` is also the better primitive on its own terms: the stateful `tf.random.uniform`
reads a GLOBAL seed another library can move, so a `seed:` knob would not actually reproduce.

**Example.**

```python
def _glorot_uniform(shape, seed):
    receptive_field = 1
    for dim in shape[:-2]:
        receptive_field *= int(dim)
    if len(shape) == 2:                       # a Dense kernel (in, out) — no receptive field
        fan_in, fan_out = int(shape[0]), int(shape[1])
    else:                                     # a conv kernel (..., in, out)
        fan_in, fan_out = int(shape[-2]) * receptive_field, int(shape[-1]) * receptive_field
    limit = (6.0 / float(fan_in + fan_out)) ** 0.5
    return tf.random.stateless_uniform(shape, seed=(seed, 0), minval=-limit, maxval=limit)
```

**What you may change.** The initializer, if the spec's comparability is preserved — the parameter
COUNT is what `spec.trainable_parameter_count` pins, not the distribution. What you may not do is
reach back into `tf.initializers` for brevity.

---

## 11. One name per quantity, chosen where the name is chosen (2026-08-04)

**Context.** `examples/sweep/run_sweep.sh` trains ONE config on every installed
backend so the runs can be compared in a single table. They could not be. Every
tracker records the key it is handed, so each run landed its engine's own spelling —
measured from the header line of each `metrics.csv` in a real 13-run sweep:

```
lightning     train/loss_epoch  train/accuracy_top1  val/loss  val/accuracy_top1
torch         train/loss        train/accuracy_top1  val/loss  val/accuracy_top1
fastai        train_loss        valid_loss           accuracy  top_k_accuracy
keras         loss              accuracy             val_loss  val_accuracy
tensorflow    loss              sparse_categorical_accuracy    val_loss  val_sparse_categorical_accuracy
huggingface   train_loss        eval_loss            eval_accuracy
jax / mlx     loss              val_loss             val_accuracy
```

The report printed the union of those keys: one quantity in four columns, every row
mostly blank, and the report said so in its own source ("the backends genuinely
disagree on naming"). That report is now `matrainer.sweep`.

**Decision.** The fix is split by WHERE each name is decided, and the split is the
whole point — neither half can do the other's job.

*The split prefix and the loss* are decided by the engine, which sonair does not
control, so they are TRANSLATED at the boundary: each backend hands what it reports
through `matrainer.MetricNames` (record 6 in matrainer's own architecture doc),
declaring its own `source`, on the way to a tracker and to the metrics sidecar. Only
the backend knows which engine wrote the names and which split an unprefixed key
belongs to — fastai's bare `accuracy` is a VALIDATION metric where Keras' is a
training one.

*The metric's own name* is decided by whoever wires the metric, so it is fixed THERE
rather than mapped: the torch/Lightning defaults and the example config say
`accuracy` instead of `accuracy_top1`, and the Keras/TensorFlow defaults pass
`name="accuracy"` to the metric object instead of letting `compile()` expand the
`"accuracy"` alias into `sparse_categorical_accuracy`. The reason it goes this way
round is that the OTHER five backends cannot be moved: fastai, JAX, MLX and
HuggingFace take their metric's name from the function's `__name__`, so `accuracy`
is the only word all eight can produce. Renaming it in code instead would mean
rewriting a name the user WROTE in their config, which is a different and worse
thing than translating one an engine generated.

**Consequences.**

- Every backend now records `train/loss`, `val/loss` and `val/accuracy`. Verified by
  running the sweep: 13/13 runs, CSV and MLflow alike.
- A metric only one backend computes keeps its own column — the report shows the
  union, because those are genuinely different measurements. The sweep config no longer
  has one: `val/f1_macro` reached every engine under record 12, and fastai's
  `val/top_k_accuracy` was dropped once it was the last ragged column (a k=5 accuracy on
  10 classes, which chance alone scores 0.5 on). The union rule stays, because it is a
  property of the report rather than of one config.
- `LightningClassifier.training_step` logs the loss TWICE, under `train/loss`
  (`on_epoch`) and `train/loss_step` (`on_step`). Lightning appends `_step`/`_epoch`
  only when one name carries both, and `train/loss_epoch` is a name no other backend
  writes. The per-step series is kept — nothing else shows the loss moving within an
  epoch — and it keeps the progress bar's live number. The sweep report FOLDS it away
  by default under its own reason (`is_within_epoch_series`, not `BOOKKEEPING`): a
  one-row-per-run table cannot show a finer axis, and only this backend fills the
  column — but calling a genuine result "bookkeeping" in the footer would be a
  different kind of wrong.
- `HuggingFaceClassifier.test()` passes `metric_key_prefix="test"` to
  `Trainer.evaluate()`. Left at HF's `eval` default the held-out scores would be
  translated into the `val/` columns a validation pass fills.
- Every backend gained a `metric_names:` constructor slot, so one top-level key
  reaches all eight — including `target:`, which records the whole sweep in another
  engine's spelling.
- What an engine hands its OWN callbacks is untouched: `EarlyStopping(monitor="val_loss")`
  still finds Keras' name, and `KerasClassifier._history` keeps Keras' columns while
  `computed_metrics` carries the translated ones.

**Example.**

```yaml
metrics:
  accuracy: !lazy:torchmetrics.classification.MulticlassAccuracy { top_k: 1 }

metric_names: !class:matrainer.MetricNames    # optional; the default is <split>/<name>
  target: keras
```

```
| run           | val/accuracy | val/f1_macro | val/loss | train/loss |
| keras_zoo     |       0.69   |       0.6612 |   0.9272 |     0.7906 |
| lightning_zoo |       0.2411 |       0.1498 |   2.3028 |     1.1786 |
```

**What you may change.** A new backend declares the slot and calls
`as_metric_names(self.metric_names)` where it reports (pinned fleet-wide by
`tests/test_tracking_parity.py`). Name a new metric with the word every engine can
produce. Do NOT translate a name the user wrote in a config, and do not push the
translation down into `Tracker.log_metrics` — a tracker cannot know which engine or
which split produced a bare key.


## 12. Naming a quantity is half the job; every engine must be able to COMPUTE it (2026-08-04)

**Context.** Record 11 made the eight backends agree on what a quantity is CALLED, and the sweep
then showed the other half of the problem: the columns were consistent and mostly empty.
`val/f1_macro` reached two backends of eight, the training columns three, so eleven of thirteen
runs were blank in three columns of a table whose entire purpose is comparison. Nothing was
misnamed. The engines simply could not produce the measurement:

- Keras publishes `F1Score`, but it compares two `[N, C]` tensors while a single-label run here
  produces int64 class ids `[N]` — wired as it stands it dies on the first batch.
- `fastai.metrics` publishes `F1Score` as a FACTORY, so the only thing a config can name resolves
  to the factory itself.
- The JAX, MLX and HuggingFace metric sets held two functions each, neither of them an F1.
- Three backends computed no training metrics at all, and a fourth (Lightning) computed them
  per-STEP, which `log_every_n_steps` then discarded on any run shorter than 50 steps.

**Decision.** Give each engine family the smallest thing that lets it compute the SAME quantity,
and put the arithmetic that is genuinely shared upstream.

- `keras` / `tensorflow`: **`sonair.keras_classification.SparseF1Score`** — the one loss/metric
  class this project owns, and the exception record 5's rule already allows (a transport shim for
  something unregisterable as it stands). It one-hots `y_true` and delegates; no F1 arithmetic of
  its own. Its name is DERIVED as `f1_<average>`, so `average="micro"` cannot be recorded under
  the macro name.
- `fastai`: **configured aliases** (`f1_macro` / `f1_micro` / `f1_weighted`) resolving to a BUILT
  and renamed `F1Score(average=…)`.
- `jax` / `mlx`: a **`_CONFUSION_METRICS`** table beside the per-batch `_METRICS`, partitioned by
  NAME before resolution.
- `huggingface`: a numpy `_f1_macro` — its hook already sees the whole split at once.
- The counting and the reduction are ONE implementation in `matrainer.metric_functions` (its
  architecture record 7), because three engines need identical arithmetic.

**Consequences.**

- **A per-class metric may not be averaged over batches.** This is the part that is invisible
  when wrong: batch-averaged macro-F1 reads three to five points LOW (0.8849 where the truth is
  0.9182, on 10 classes at batch 32) and always downward, because a class absent from a batch
  scores 0 for that batch. In THIS project's sweep that would have been a systematic penalty on
  exactly the backends that own their loop — the one bias a cross-framework comparison must not
  introduce. So a confusion metric accumulates one `C×C` matrix over the split and reduces once,
  which is what `torchmetrics` does internally.
- **Only a NAMED metric can be confusion-accumulated.** The partition happens before resolution,
  because nothing here can know what a user-supplied callable would want accumulated.
- **A multi-label run DROPS a confusion metric with a warning.** A multi-hot row has no single
  cell to land in, and a number under the name `f1_macro` computed some other way is worse than a
  gap — nothing downstream could tell.
- **`train_metrics` reached five more backends, and both blocking claims were about THIS code.**
  "The JAX/MLX training passes compute the loss and nothing else" described the loop as written;
  the step now returns its logits as an aux output. "A fastai `Learner` computes its metrics on
  the validation pass" is `Recorder`'s DEFAULT, not a constraint. Neither backend runs a second
  forward pass — that would cost a pass AND measure a different network, since `training=True`
  leaves dropout and batch norm in train mode.
- **`LightningClassifier` moved its train metrics to the epoch end**, matching every other
  backend and its own eval splits. Only `huggingface` still has no training metrics, and that one
  IS the engine: `compute_metrics` is its evaluation hook.
- A training metric therefore measures the network as it trained, so a gap against validation is
  expected. The bare-torch backend's BatchNorm note in `AGENTS.md` is the same observation.

**Example.**

```yaml
keras: !scope:framework=keras
  metrics:
    - !class:keras.metrics.SparseCategoricalAccuracy { name: accuracy }
    - !class:sonair.SparseF1Score                      # -> "f1_macro"

jax: !scope:framework=jax
  runnable: !class:sonair.JaxClassifier
    metrics: [accuracy, f1_macro]        # f1_macro is ACCUMULATED, accuracy per batch
    train_metrics: [accuracy, f1_macro]  # no second forward pass — the step returns its logits
```

**What you may change.** Add a metric derived from a confusion matrix (per-class recall, kappa) —
the accumulator is already there, so it is a reduction in `matrainer.metric_functions` plus a name
in each backend's table. Add an engine's own metric where it ships one. Do NOT average a per-class
metric over batches to avoid the accumulator, do NOT approximate a confusion metric for a
multi-label run, and do NOT re-run the model in eval mode to produce training metrics.

---

## 13. The Lightning backend runs on the workspace trainer bases (2026-08-06)

**Context.** The four task projects were a four-way mirror: the same merged
train/evaluate/test/predict runnable per engine, with the Lightning modules 40–61%
byte-identical ACROSS projects (tracker wiring, trainer construction, step skeletons, epoch
reporting, sidecars, outputs) and the task logic near-identical WITHIN each project across its
engines. Both projects' own comments called the duplication deliberate-and-temporary, deferred
until the fleet existed to design a base against. By 2026-08-06 it did.

**Decision.** The trainer machinery moved UP to matrainer (its `docs/architecture.md` record 9
carries the full shape): a neutral abstract API (`TrainerRunnable`), a torch family layer
pinning the types (`TorchTrainerRunnable`), and the Lightning machinery base
(`LightningRunnable`). THIS project keeps only its task half, written once for the whole engine
fleet — `classification_base.py` (torch-free, importable by every backend) + `classification_torch.py` (the
torch-typed hook implementations) — and the concrete Lightning class becomes the composition
plus its own full flat constructor (the confluid introspection surface, signature unchanged).

**Consequences.** One task behaviour = one hook implementation instead of one copy per engine;
the fleet drift tests now check "inherits the shared base and does not shadow its wiring"
instead of per-module source patterns; matrainer is load-bearing for training (the old
"removing it keeps the class training" claim is dead by design). The other engine backends
joined family by family through 2026-08-08, when the migration COMPLETED — every backend
now composes its task hooks with a shared machinery base (see the closing amendment below).

**Example.**

```python
@configurable(task="classification", role="trainer", framework="torch")
class LightningClassifier(ClassificationTorchHooks, LightningRunnable):
    def __init__(self, model, train_set=None, ..., label_map=None):
        super().__init__(model=model, train_set=train_set, ...)  # shared slots
        self.label_map = label_map                                          # task-specific state
```

**Amendment (2026-08-07).** The bare-torch backend joined the same shape: ``TorchClassifier`` is `(ClassificationTorchHooks, matrainer.torch.TorchLoopRunnable)` — the same task hooks, a different machinery base (the owned epoch loop instead of a `pl.Trainer`), plus the `_after_fit` epilogue saving the class names beside `model.pt`. Its `evaluate` was harmonized to score the VAL split (it used to delegate to `test()`).

**Amendment (2026-08-07, second).** The fastai backend joined: ``FastaiClassifier`` is `(ClassificationTorchHooks, matrainer.fastai.FastaiRunnable)` — the same task hooks, the Learner machinery base, plus the task's fastai vocabulary as hooks (`_default_fastai_loss` — the `*Flat` pair, `_default_fastai_metrics`, `_metric_aliases` — the `f1_*` configured forms) and the `_after_fit` class-names epilogue. Its `evaluate` was harmonized to score the VAL split.

**Amendment (2026-08-07/08, closing — P5 + P6, the migration completes).** The remaining five
backends joined and the fleet is DONE. P5 (keras family): ``KerasClassifier`` and
``TensorFlowClassifier`` are ``(ClassificationKerasHooks, matrainer.keras.KerasRunnable)`` and
``(…, matrainer.tensorflow.TensorFlowRunnable)`` — ONE hooks module (`classification_keras.py`)
serving both machinery siblings, which also fixed the TF loop sharing one metric accumulator
across splits and the missing fork-warm. P6: ``HuggingFaceClassifier`` is
``(ClassificationBase, matrainer.huggingface.HuggingFaceRunnable)`` (the label pipeline is the
task base's; the `problem_type` placement, the multilabel collate and the numpy metric names
stay here), and ``JaxClassifier`` / ``MlxClassifier`` are ``(ClassificationBase,
matrainer.jax.JaxLoopRunnable)`` / ``(…, matrainer.mlx.MlxLoopRunnable)`` with the task's
accumulation vocabulary as the machinery's split-state hooks (`f1_macro` over one confusion
matrix per split). Every `evaluate` scores the VAL split, fleet-wide. What remains task-owned
is exactly the hooks: the batch dtypes, the loss/metric defaults by target shape, and the
class-names artifact.

**What you may change.** Hook implementations and task state freely; the concrete signature
(it is the config surface). What you may not do: shadow the inherited machinery
(`_build_trainer` / `_live_trackers` / `_run_eval` / `_report_epoch_metrics`) — extend through
the hooks, or change the base in matrainer with every consumer updated in the same workspace
change.
