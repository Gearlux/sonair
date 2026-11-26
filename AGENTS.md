# Sonair Mandates

Rules only. The reasons are in [`docs/architecture.md`](docs/architecture.md) (cited as §N),
and usage is in the [README](README.md) and [`docs/frameworks.md`](docs/frameworks.md). Root
`AGENTS.md` rules are not repeated here.

## Current state

Image classification on eight backends, all driven by one config: `lightning`, `torch` (a
hand-written loop), `keras`, `tensorflow`, `fastai`, `huggingface`, `jax`, `mlx`. On PyPI as
pre-release `0.1.0a1`; the repo is private.

Each runnable `<backend>_classifier.py` = task hooks (here) + a matrainer machinery base + its
own full flat `__init__`:

| backends | task hooks | matrainer base |
|---|---|---|
| lightning · torch · fastai | `ClassificationTorchHooks` | `LightningRunnable` · `TorchLoopRunnable` · `FastaiRunnable` |
| keras · tensorflow | `ClassificationKerasHooks` | `KerasRunnable` · `TensorFlowRunnable` |
| huggingface · jax · mlx | `ClassificationBase` | `HuggingFaceRunnable` · `JaxLoopRunnable` · `MlxLoopRunnable` |

Other modules:
- `classification_base.py` is torch-free: label pipeline, `num_classes`, `class_names`.
- `classification_torch.py` / `classification_keras.py`: the per-family hooks.
- Zoo wrappers: `timm_model.py`, `keras_model.py`, `tensorflow_model.py`,
  `huggingface_model.py`.
- `convnet/`: `spec.py` plus five implementations of it.
- `{torch,keras,fastai}_classification.py`: registered upstream losses and metrics.
- `image_folder.py`: the folder-per-class reader and writer.
- `cli.py`: the `FRAMEWORKS` table only, built on `matrainer.cli`.
- `__init__.py`: lazy exports.

Example config: `examples/train_efficientvit_b0_mnist.yaml`. Sweep: `examples/sweep/`.

**Who owns what:**
- recordstream: records, batching, collate, `LabelMap` and its file format.
- matrainer: trainer machinery, tracking, artifacts, metric naming and merging, the CLI factory.
- sonair: the classification hooks, models, conv spec, registrations and configs.

Never add a filesystem, vocabulary, batching, collate or CLI helper here. For any new helper,
first ask which package owns its vocabulary.

## Backends

- Keep each concrete backend's FULL flat `__init__`, because introspection reads it. No
  `**kwargs`. (`tests/test_tensorflow_classifier.py::test_the_constructor_mirrors_the_keras_backends_knobs`)
- Extend through the hooks. Never shadow the machinery (`_build_trainer`, `_run_eval`,
  `_report_epoch_metrics`, `fit`, `_score`, `_epoch_metrics`, `_log_run_context`,
  `_live_trackers`), and never derive hyperparameters privately. A machinery change goes into
  matrainer, with every consumer updated. (`tests/test_tracking_parity.py`) §13.
- A new backend is added to all of: `FRAMEWORKS`, an extra, an entry point,
  `EXPECTED_FRAMEWORKS`, `BACKENDS`/`METRIC_BRIDGES` (tracking parity), `SERVED` (metric
  slots). The lists are checked against `*_classifier.py` on disk. Recipe: `docs/frameworks.md`.
- `TorchClassifier` imports no training framework
  (`tests/test_torch_classifier.py::test_it_needs_no_training_framework`). §14.
- `TensorFlowClassifier` is a sibling of `KerasClassifier`, never a subclass (§9). It imports
  keras/tf only via `sonair._tensorflow` (`tests/test_tensorflow_classifier.py`). Both must agree
  on the classification half
  (`::test_the_duplicated_classification_half_agrees_with_the_keras_backend`).
- `huggingface` has no loss set, no convnet and no `train_metrics` (TASKS.md). Its test fakes
  return a real `ImageClassifierOutput`, because `Trainer` subscripts the output.
- `fastai`, `jax` and `mlx` take losses, metrics and optimizers by NAME. Never add a wrapper
  class to make them registrable.

## Config and CLI

- The backend is a flag: `--framework X` → `!scope:framework=X`. No backend subcommands. No
  CLI machinery in `cli.py`; it lives in `matrainer.cli`. §1b.
- `runnable`, `model`, `metrics` and `preprocess` are TOP-LEVEL siblings. `runnable:` sits
  inside each `!scope:framework=` block, and `default_scopes: [framework=lightning]` picks the
  default. Never nest `model:` in `runnable:`.
  (`tests/test_cli_frameworks.py::test_every_declared_framework_resolves_to_a_complete_runnable`)
- A non-torch block sets `metrics: null` and `val_metrics: null`
  (`::test_no_framework_inherits_a_metric_set_its_engine_cannot_consume`).
- Each scope block keeps its `import: sonair.<engine>_classification` line.
- A config feeds the model through `Stream(source=…, ops=!ref:preprocess)`, because the HF
  source does not tensorize.
  - Define `preprocess` once and reuse it for both splits.
  - Torch chains end in `ToTensor`.
  - Keras chains stop before it, and do not normalize for `keras.applications`.
- A loader kwarg that `loader_slots` bakes (`persistent_workers`) needs a constructor knob. An
  unbaked one (`multiprocessing_context`) arrives by broadcasting.
- Keep the five extra-gated example blocks commented out (`tests/test_examples_load.py`). The
  sweep overlay is where they run.

## Registrations and discovery

- Register upstream classes; never reimplement one. The only exceptions are shims for things
  that cannot be registered as they are: `SigmoidFocalLoss` and `SparseF1Score`. §2.
- One registration module per engine, each with its own entry point. There is no
  `tensorflow_classification.py`. §2, §4.
- New loss/metric: add it to the `_CLASSIFICATION_*` tuple AND the literal `__all__`, then
  `uv pip install -e ./sonair --no-deps`. (`tests/test_categories.py`,
  `tests/test_keras_classification.py`)
- Every registering module gets its entry point in the SAME change. Entries are modules, never
  `module:attr`. (`tests/test_entry_points.py`)
- Every class carries `task`, `role` and `framework` tags. `framework` is the API the class is
  wired to: the TensorFlow classes are `keras` and the fastai losses are `torch`.
  (`tests/test_categories.py`)
- torchmetrics register with `broadcast="declared"`, and `classification_torch.py` imports
  `sonair.torch_classification` ON PURPOSE (`tests/test_metric_broadcast.py`). The keras gap is
  in TASKS.md.
- Registration modules import their framework at module level ON PURPOSE, so a missing engine
  drops its whole set.
- Keras: import it via `sonair._keras` or `recordstream.keras` and use `keras.losses.X`; never
  `import keras` / `from keras.losses import X`. Every crossentropy on this logits head takes
  `from_logits=True`.
  (`tests/test_keras_classifier.py::test_the_default_loss_reads_its_input_as_logits`)
- A new constructor param on any classifier makes navigaitor's examples stale. Run
  `python -m navigaitor.scripts.regen_examples` in the same change, and revert files where only
  the date stamp changed.

## Task pipeline

- A sequence as the first target means multi-label. There is no `multilabel:` knob. §3.
- Train targets are walked AT MOST ONCE, for the label fit, `num_classes` and the weights.
  *(unpinned)*
- A concrete model's head is authoritative for `num_classes`, and eval reads the count off the
  model. A `LabelMap` is fitted on train only and reloaded on eval, never refit.
  (`tests/test_classification_trainer.py::test_concrete_model_num_classes_is_authoritative`,
  `tests/test_classification_evaluator.py::test_string_targets_encoded_via_checkpoint_class_names`)
- `loss_fn`/`metrics` default to `None` and resolve before the balancing probe. A wired one is
  never replaced. (`::test_an_explicitly_wired_loss_is_never_replaced`) §3.
- Class weights: recordstream computes the statistic, and the family hook
  `apply_class_weights()` injects it. (`::test_the_weight_STATISTIC_comes_from_recordstream`,
  `::test_balancing_hooks_are_overridable_per_framework`)
- There is no sonair collate. `_targets` picks float multi-hot or explicit int64
  (`::test_the_classifier_names_int64_for_single_label_targets`). A new target shape is a
  recordstream primitive plus a branch here.
- Predictions are `ClassificationOutput`, never raw logits (`tests/test_predictions.py`).
- Call bare `flow(x)` without an `isinstance(x, Fluid)` guard, and flow the model
  unconditionally. Keep a guard only to read `.target`/`.kwargs`, to pass runtime kwargs, or to
  catch a bare `type`.
  (`tests/test_keras_classifier.py::test_a_live_model_is_left_exactly_as_it_arrived`)

## Slots, metrics, outputs

- Every slot flowed at run time is `Partial[T]`: never `Any`, never `Union[X, Fluid]`. A
  `model` slot names its framework's module type, or a union alias where two types meet.
  (`tests/test_lazy_slots.py`) §8.
- `trackers` is a constructor param, `Optional[Union[Partial[Tracker],
  Iterable[Partial[Tracker]]]]`, stored verbatim. There is no `logger:` slot. `callbacks` is
  `Optional[List[...]]`, not `Iterable`. (`tests/test_tracking_parity.py`) §16.
- Every backend declares `metric_names`. Use the metric name every engine can produce
  (`accuracy`, `f1_macro`), and never rename one a user wrote. §11.
- Combine per-split metrics only via `matrainer.metric_slots`. Who serves which split is
  `tests/test_metric_slots.py::SERVED`, checked in both directions. Never re-run the model in
  eval mode for train metrics. §12.
- `trained_model`, `computed_metrics` and `class_names` are read-only `@property @output`s that
  raise before a run. `model` is `Mandatory`. (`tests/test_classification_trainer.py`,
  `tests/test_classification_evaluator.py`)

## Models

- A wrapper subclasses its engine's model base. Never write
  `getattr(self.model, "model", self.model)`. §7.
- `TimmModel`:
  - No work in `__init__`.
  - It builds once into the public read-only `classifier`, a property over
    `_modules["classifier"]`.
  - `solidify()`, `backbone` and `forward` share one cache; keep `solidify()`.
  - It forwards its knobs to `timm.create_model` unchanged.

  (`tests/test_timm_classifier.py`, `tests/test_final_config_roundtrip.py`)
- `solidify()` runs more than once per run, so every model must build once and cache.
  (`::test_flowing_a_LIVE_model_builds_it`)
- A wrapper with a random build seeds at build time and does not restore the RNG.
  `TensorFlowModel` has no seed. (`::test_the_seed_actually_reproduces_the_weights` and its keras
  and huggingface twins) §15.
- `model_refs` reads stored config only. It returns `[]` when nothing external was loaded, and
  builds entries with `hf_model_ref`/`file_model_ref`. (`tests/test_model_refs.py`)
- `convnet/`:
  - Each implementation is `<framework>_convnet.py` → `*ConvNet`, and IS its framework's module
    type.
  - All share one knob set, including `seed`.
  - All read `input_shape` as `(H, W, C)`.
  - `__init__.py` imports nothing.
  - `huggingface` gets no convnet.

  (`tests/test_convnet_spec.py`, `tests/test_torch_convnet.py`) §5‡.
- A raw-`tf.nn` module never uses `tf.initializers`, `tf.keras`, `tf.losses`, `tf.metrics` or
  `tf.optimizers`. (`tests/test_tensorflow_convnet.py`) §10.

## Datasets

- The folder-per-class format lives here, because it is a classification convention.
  `ImageFolderSource` forwards the whole source protocol, `project` included.
  (`tests/test_image_folder.py::TestTheDelegateKeepsItsReadersEfficiency`)
- `ImageFolderSink(delete_original=…)` keeps all five guards. (`::TestDeleteOriginalIsAMove`)
  - Off by default.
  - Delete the original only after the new file exists.
  - Never delete when source and destination are the same file.
  - If the original cannot be removed, the write still stands.
  - A record with no source path is written without moving anything.

## Packaging, release, tests

- `pip install sonair` installs no framework. Each Keras-family extra names its own engine.
  Keep `_BACKEND_NAMES` in step with the extras. (`release.yml` smoke test;
  `tests/test_cli_frameworks.py::test_every_framework_names_a_declared_pip_extra`)
- `fastcore<2` and `fastprogress<1.1` are upstream breaks, not hardcoded versions; see
  `pyproject.toml` and TASKS.md.
- Releases go through the hand-written `release.yml` from a private repo.
  - Each release raises every workspace floor, core and extras, to the newest release and runs the suite there,
    and a dependent's next release raises its `sonair>=` the same way (user choice 2026-11-26; aisland
    `tests/test_workspace_dependency_specs.py` checks each is satisfiable).
  - README links are absolute GitHub URLs (`tests/test_docs_links.py`).
  - The newest CHANGELOG entry is the declared version (`tests/test_packaging.py`).
- CI runs `mypy .` here with only its own extras. List every extra-gated import in
  `[[tool.mypy.overrides]]`. `--cov-branch` is required (`tests/test_coverage_config.py`).
- TF-engine tests skip in the default run. Run them with
  `KERAS_BACKEND=tensorflow pytest tests/test_tensorflow_*.py tests/test_convnet_spec.py`. CI
  does NOT run them. Never add a module-level `importorskip` to those files. §6.
- Leave the TF-first import in `tests/conftest.py`: don't delete it or make it
  unconditional, and keep `multi_label=True` in `test_auc_is_the_metric_that_carries_from_logits`.
  The conftest docstring explains why.
- On a default checkout the fastai, huggingface, jax, mlx and tensorflow suites skip. Install
  the extra before trusting a change there.

## Sweep

- `examples/sweep/sweep.yaml` is the 13-cell matrix and `all_backends.yaml` its overlay. The
  driver is `matrainer sweep backends|trackers|all`. (`tests/test_sweep_config.py`)
- In the overlay:
  - Restate a framework block in full, because `include:` does not deep-merge scope blocks.
  - Nest the per-engine `!scope:model=convnet` inside its framework block.
  - Tracker blocks use `!ref:experiment_name` / `!ref:run_name`.
  - Ask only for metrics every engine can produce.
