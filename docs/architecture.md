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

---

## 2. The framework axis: an explicit tag, with MRO inference as fallback

*2026-07-28, implemented 2026-07-29*

**Context.** [`classification.py`](../sonair/classification.py) registers
off-the-shelf losses and metrics rather than wrapping them —
`torch.nn.CrossEntropyLoss` and the `torchmetrics.classification.Multiclass*`
family. They were tagged `task="classification", role="loss"` / `role="metric"`,
which say **what a class is for** but never **which engine it runs on**.

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

**Example.**

```python
# tag at registration — classes and builder functions alike
register(nn.CrossEntropyLoss, task="classification", role="loss", framework="torch")

# the picker query that is now answerable
list_classes(task="classification", role="loss", framework="keras")   # -> no torch losses
```

**What you may change.** Tag a class's framework whenever it is engine-bound;
leave it off when it genuinely is not (a pure-python op, a path-based source).
Do not fold framework into `category`, and do not make an untagged class
excluded-by-default — the opt-in asymmetry is what let this ship without
re-tagging the workspace.

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
