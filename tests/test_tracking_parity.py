"""Every backend answers "what produced this run?" the same way.

Two of the eight used to transcribe their own configuration into a params dict and FOUR
captured no provenance at all — `trackers:` was wired but only ever received the class-name
artifact. That is invisible from a passing test suite and from a green run: the tracker is
present, the run completes, and the record is simply thin.

These tests read the SOURCE rather than importing, because no venv has every backend's
framework installed — an import-based check would silently reduce to whichever extras are
present locally, which is exactly the machine where a copy-pasted omission lands.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, Set

import pytest

SOURCE_DIR = Path(__file__).resolve().parent.parent / "sonair"

#: Every backend runnable, by module stem. A new one must be added here — which is the point.
BACKENDS = [
    "lightning_classifier",
    "torch_classifier",
    "keras_classifier",
    "tensorflow_classifier",
    "fastai_classifier",
    "jax_classifier",
    "mlx_classifier",
    "huggingface_classifier",
]


def built_on_shared_base(source: str) -> bool:
    """The 2026-08-06 consolidation: a backend on a shared machinery base inherits its
    tracking wiring — `runnable=self` + provenance via `TrackingCallback` in `_build_trainer`
    (lightning) / `_log_run_context` + the per-epoch `tracker.log_metrics` and terminal
    `capture_log_file` in `fit` (the owned loop), metric forwarding, the naming knob's
    application — all pinned in matrainer's own suite. What such a module must not do is
    SHADOW that wiring."""
    return "LightningRunnable" in source or "TorchLoopRunnable" in source


def assert_does_not_shadow_the_tracking(stem: str, source: str) -> None:
    shared = (
        # the Lightning machinery base's wiring
        "_build_trainer",
        "_run_eval",
        "_report_epoch_metrics",
        # the owned-loop machinery base's wiring
        "fit",
        "_score",
        "_epoch_metrics",
        "_log_run_context",
        # both
        "_live_trackers",
    )
    for name in shared:
        assert f"def {name}" not in source, (
            f"{stem} is built on the shared base but shadows {name} — the inherited tracking "
            "wiring no longer applies; extend via the hooks instead."
        )


def module_source(stem: str) -> str:
    return (SOURCE_DIR / f"{stem}.py").read_text()


def calls_with_runnable_self(source: str) -> Set[str]:
    """Names of functions called with a literal ``runnable=self`` keyword."""
    found: Set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg == "runnable" and isinstance(keyword.value, ast.Name) and keyword.value.id == "self":
                target = node.func
                found.add(target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "?"))
    return found


def called_names(source: str) -> Set[str]:
    return {
        (node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", ""))
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
    }


@pytest.mark.parametrize("stem", BACKENDS)
def test_every_backend_hands_matrainer_the_runnable(stem: str) -> None:
    """`runnable=self`, never a params dict the backend transcribes for itself."""
    source = module_source(stem)
    if built_on_shared_base(source):
        assert_does_not_shadow_the_tracking(stem, source)
        return
    entries = calls_with_runnable_self(source)
    assert entries, (
        f"{stem} does not pass `runnable=self` to any tracking entry point. "
        "Hand matrainer the object so the hyperparameters come from its one derivation "
        "(provenance.config_params), not from a dict this backend writes by hand."
    )
    assert entries <= {"log_run_context", "TrackingCallback", "keras_tracking_callback"}, entries


@pytest.mark.parametrize("stem", BACKENDS)
def test_every_backend_captures_run_provenance(stem: str) -> None:
    """Wiring `trackers:` must record the run, not just its class names."""
    source = module_source(stem)
    if built_on_shared_base(source):
        assert_does_not_shadow_the_tracking(stem, source)
        return
    names = called_names(source)
    entry_points = {"log_run_context", "TrackingCallback", "keras_tracking_callback"}
    assert names & entry_points, (
        f"{stem} wires trackers but never captures run provenance — no git state, no config, "
        "no hyperparameters reach them. Call one of: " + ", ".join(sorted(entry_points))
    )


@pytest.mark.parametrize("stem", BACKENDS)
def test_every_backend_finishes_its_manifest(stem: str) -> None:
    """An unfinished manifest stays `running`, and the atexit hook then calls it interrupted.

    A backend that drives the plain functions must call ``capture_log_file`` itself; the two
    that hand tracking to a callback get it from the callback's end hook instead.

    Since 2026-08-02 the same call also ENDS the run on each tracker, so this one reachability
    check covers both halves of "the run is over" — see
    ``test_the_terminal_call_is_the_one_that_ends_the_run`` below for why they were folded
    together rather than given a call each.
    """
    source = module_source(stem)
    if built_on_shared_base(source):
        # The lightning base wires TrackingCallback, whose end hook owns the lifecycle; the
        # owned-loop base's `fit()` calls `capture_log_file` itself. Both pinned upstream.
        return
    names = called_names(source)
    if names & {"TrackingCallback", "keras_tracking_callback"}:
        return  # the callback owns the lifecycle
    assert "capture_log_file" in names, (
        f"{stem} starts a manifest via log_run_context but never finishes it; call "
        "capture_log_file(trackers, manifest=..., status='success') at the end of fit()."
    )


def test_the_terminal_call_is_the_one_that_ends_the_run() -> None:
    """No backend calls ``Tracker.finalize`` itself, and none should — matrainer does it.

    The gap this pins was found by a real MLflow sweep on 2026-08-02: eleven of thirteen
    FINISHED runs showed ``RUNNING`` with no end time, because nothing anywhere called
    ``finalize``. The two that looked right did so by accident — a Lightning consumer adapts
    its trackers into ``pl.Logger``s and Lightning finalizes those itself.

    The fix went into ``provenance.capture_log_file``, which already stated the run's terminal
    status and already looped over the trackers — so every backend got it at once. A per-backend
    call would be eight chances to forget and eight places for the status to disagree with the
    manifest's, which is exactly the divergence the tracking-parity rules exist to prevent.
    """
    offenders = {stem for stem in BACKENDS if "finalize" in called_names(module_source(stem))}

    assert not offenders, (
        f"{sorted(offenders)} call Tracker.finalize directly. Ending the run belongs to "
        "matrainer's capture_log_file, which every backend already reaches — a local call would "
        "let a backend's terminal status disagree with its manifest's."
    )


def test_no_backend_keeps_a_private_hyperparameter_derivation() -> None:
    """`_tracked_params` / `_config_params` are the two spellings that diverged before."""
    offenders: Dict[str, List[str]] = {}
    for stem in BACKENDS:
        methods = [
            node.name
            for node in ast.walk(ast.parse(module_source(stem)))
            if isinstance(node, ast.FunctionDef) and node.name in ("_tracked_params", "_config_params")
        ]
        if methods:
            offenders[stem] = methods
    assert not offenders, f"private hyperparameter derivations are back: {offenders}"


def test_the_backend_list_covers_every_runnable_module() -> None:
    """A ninth backend must not be able to skip these checks by not being listed."""
    on_disk = {
        path.stem
        for path in SOURCE_DIR.glob("*_classifier.py")
        if "class ClassifierTask" not in path.read_text()  # exclude non-runnable helpers
    }
    assert on_disk == set(BACKENDS), f"unlisted backend module(s): {on_disk ^ set(BACKENDS)}"


@pytest.mark.parametrize("stem", BACKENDS)
def test_every_backend_forwards_metrics_to_trackers(stem: str) -> None:
    """Provenance without curves is half a record — a wired tracker must see the metrics.

    Four of the eight recorded the config and never called `log_metrics`, so an MLflow run
    showed the hyperparameters and an empty chart. The three routes are all legitimate: the
    loop calls `log_metrics` itself, a matrainer callback forwards `on_epoch_end`, or the
    framework's own history is replayed after the fit.
    """
    source = module_source(stem)
    if built_on_shared_base(source):
        assert_does_not_shadow_the_tracking(stem, source)
        return
    names = called_names(source)
    forwards = "log_metrics" in names
    via_callback = bool(names & {"TrackingCallback", "keras_tracking_callback"})
    replays = bool(names & {"_log_recorded_epochs", "_log_trainer_history"})
    assert forwards or via_callback or replays, (
        f"{stem} never forwards metrics to its trackers — a wired backend would record the "
        "config and draw an empty curve."
    )


@pytest.mark.parametrize("stem", BACKENDS)
def test_every_backend_takes_the_metric_naming_knob(stem: str) -> None:
    """One `metric_names:` key must reach ALL EIGHT, or a sweep's columns stop lining up.

    The failure it prevents was measured across a real eight-backend sweep: each engine names
    the same quantity its own way (`val_loss` under Keras, `valid_loss` under fastai,
    `eval_loss` under a `transformers.Trainer`, `val/loss` under torch), and nothing renamed
    them on the way out — every tracker writes the key it is handed. One backend left out of
    this puts its curves in columns of their own.

    Read from the ANNOTATION for the reason the rest of this file reads source: no venv has
    every engine, so an import-based sweep would skip exactly where a copy-pasted omission lands.
    """
    for node in ast.walk(ast.parse(module_source(stem))):
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            annotations = {arg.arg: ast.unparse(arg.annotation) for arg in node.args.args if arg.annotation is not None}
            assert "metric_names" in annotations, f"{stem} cannot be told how to name what it records"
            assert "Lazy[MetricNames]" in annotations["metric_names"], (
                f"{stem}: the slot must be `Optional[Lazy[MetricNames]]` — wired from YAML it "
                f"arrives as a deferred marker (got {annotations['metric_names']})"
            )


@pytest.mark.parametrize("stem", BACKENDS)
def test_every_backend_applies_the_naming_it_was_given(stem: str) -> None:
    """Declaring the slot is half of it — the metrics it reports have to go THROUGH it."""
    source = module_source(stem)
    if built_on_shared_base(source):
        assert_does_not_shadow_the_tracking(stem, source)
        return
    names = called_names(source)

    assert "as_metric_names" in names, (
        f"{stem} declares `metric_names` but never applies it: call "
        "`as_metric_names(self.metric_names)` where metrics reach a tracker or the sidecar."
    )


CALLBACK_BACKENDS = [
    "keras_classifier",
    "tensorflow_classifier",
    "fastai_classifier",
    "jax_classifier",
    "mlx_classifier",
    "huggingface_classifier",
]


@pytest.mark.parametrize("stem", BACKENDS)
def test_no_backend_keeps_a_logger_slot(stem: str) -> None:
    """`logger:` named two different things and neither survived.

    Lightning's held `pytorch_lightning.loggers` (now the trackers, adapted); the other six
    held framework CALLBACKS under a misleading name (now `callbacks:`).
    """
    for node in ast.walk(ast.parse(module_source(stem))):
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            names = {arg.arg for arg in node.args.args + node.args.kwonlyargs}
            assert "logger" not in names, f"{stem} still declares a `logger` constructor slot"
            assert "trackers" in names


@pytest.mark.parametrize("stem", CALLBACK_BACKENDS)
def test_a_list_valued_callbacks_key_reaches_the_slot(stem: str) -> None:
    """The same annotation gate `trackers` needed: a LIST only lands on a list-admitting slot.

    Checked on the ANNOTATION rather than by building each backend, because most need a
    framework a default install lacks — and a bare attribute assigned post-construction is
    exactly the shape that silently swallowed the key before.

    `List` and `Iterable` both satisfy confluid's gate (measured against
    `engine._classify_annotation`, which reports `list` for either), but only `List[X]`
    carries `X` through `to_pydantic` — an `Iterable[X]` field converts to a bare `Any`
    whatever `X` is, so the element type reaches no form spec. That is why these slots are
    spelled `List[...]` while `trackers` keeps its `Union[Lazy[T], Iterable[Lazy[T]]]` shape:
    there the single-tracker arm is what carries the type.
    """
    for node in ast.walk(ast.parse(module_source(stem))):
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            annotations = {arg.arg: ast.unparse(arg.annotation) for arg in node.args.args if arg.annotation is not None}
            assert "callbacks" in annotations, f"{stem} cannot wire framework callbacks from config"
            assert any(c in annotations["callbacks"] for c in ("List", "Iterable")), (
                f"{stem}: `callbacks` must admit a list, or confluid will not broadcast a "
                f"list-valued key into it (got {annotations['callbacks']})"
            )


def _statement_order(stem: str, names: Set[str]) -> Dict[str, int]:
    """Line number of the first call to each of ``names`` inside the module's ``fit``."""
    tree = ast.parse(module_source(stem))
    fit = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "fit"),
        None,
    )
    assert fit is not None, f"{stem}: no fit() found"
    found: Dict[str, int] = {}
    for node in ast.walk(fit):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        if name in names:
            found.setdefault(name, node.lineno)
            found[name] = min(found[name], node.lineno)
    return found


@pytest.mark.parametrize("stem", ["keras_classifier", "tensorflow_classifier"])
def test_no_tracker_write_is_sequenced_after_the_run_ends(stem: str) -> None:
    """The two callback-driven backends must not log an artifact after their run is closed.

    `keras_tracking_callback`'s `on_train_end` closes the run out, and since 2026-08-02 that
    means ENDING it on every tracker. Both backends used to call `save_class_names` — which logs
    a tracker artifact — after `fit()` returned, so the artifact arrived at a finished run.
    MLflow tolerates it (the file lands, status stays FINISHED); `WandbTracker.finish()` drops
    the run handle, so the next access starts a NEW run and the artifact lands there instead.

    Read as SOURCE ORDER rather than by running: neither backend's engine is installed on every
    machine, and the ordering is a property of the code either way.
    """
    order = _statement_order(stem, {"save_class_names", "fit", "on_train_end"})
    end_of_training = order.get("fit") or order.get("on_train_end")

    assert end_of_training is not None, f"{stem}: found neither model.fit() nor callbacks.on_train_end()"
    assert "save_class_names" in order, f"{stem}: no save_class_names call in fit()"
    assert order["save_class_names"] < end_of_training, (
        f"{stem} logs the class-names artifact at line {order['save_class_names']}, after the "
        f"training call at line {end_of_training} — by then the tracking callback has ended the run."
    )


@pytest.mark.parametrize("stem", ["fastai_classifier", "huggingface_classifier"])
def test_the_forking_backends_warm_their_splits_in_the_parent(stem: str) -> None:
    """These two run their DataLoader workers under `fork`, so a lazy first read must not be theirs.

    `import fastai` sets the multiprocessing start method to `fork` (measured; a clean interpreter
    leaves it unset, i.e. spawn on macOS), and transformers' Trainer inherits whatever is set. A
    worker that is the FIRST to touch a HuggingFace source then builds it inside a forked child,
    where `load_dataset` -> `hf_hub_download` -> `httpx.Client()` -> `urllib.request.getproxies`
    -> `_scproxy` reaches CoreFoundation, which is not fork-safe: SIGSEGV, no Python traceback,
    surfacing only as `DataLoader worker exited unexpectedly`.

    Spawning instead is NOT the fix — measured: a model on MPS cannot be shared to a spawned
    worker (`_share_filename_: only available on CPU`). Warming in the parent is, so it is pinned
    here rather than left to whoever next edits `_initialize_fit`.
    """
    assert "ensure_materialized" in called_names(module_source(stem)), (
        f"{stem} forks its DataLoader workers but never warms its splits in the parent — call "
        "recordstream.ensure_materialized() beside ensure_record_dataset() in _initialize_fit()."
    )


#: How each backend gets its METRICS onto the wired trackers — a DIFFERENT question from the
#: provenance hookup above, and the distinction is what let a real bug through: raidar's
#: `rfdetr_detector` had `log_run_context` (so the provenance table passed) while never calling
#: `tracker.log_metrics` at all, so a tracked run recorded its git state and its log file and NOT
#: its mAP. Its evaluate path had no tracker wiring whatsoever.
#:
#: The sibling restoration project found the same shape on its `transformers.Trainer` backend, and
#: only because a backend SWEEP put eleven runs in one table and one came out blank. A backend read
#: ALONE looks perfect — it prints its score and writes its metrics sidecar.
METRIC_BRIDGES = {
    "fastai_classifier.py": "log_metrics",
    "huggingface_classifier.py": "_log_trainer_history",
    "jax_classifier.py": "log_metrics",
    "keras_classifier.py": "keras_tracking_callback",
    "lightning_classifier.py": "TrackerLogger",
    "mlx_classifier.py": "log_metrics",
    "tensorflow_classifier.py": "keras_tracking_callback",
    "torch_classifier.py": "log_metrics",
}


def test_every_backend_puts_its_METRICS_on_the_trackers() -> None:
    """Provenance is not enough: a run whose numbers reach only the log is invisible to every
    comparison, every tracker UI and every sweep."""
    for name, marker in METRIC_BRIDGES.items():
        source = (SOURCE_DIR / name).read_text()
        if built_on_shared_base(source):
            # The bridge lives in the shared machinery base (the `TrackerLogger` adapters in
            # `_build_trainer` / the `tracker.log_metrics` calls in the loop's `fit` and
            # `_score`) — pinned in matrainer's own suite; no shadow = it applies.
            assert_does_not_shadow_the_tracking(name, source)
            continue
        assert marker in source, f"{name} never reaches a tracker's metric channel via {marker}"


def test_the_metric_bridge_table_covers_every_backend_on_disk() -> None:
    """The converse — a new backend must not be added without a metric channel."""
    on_disk = {p.name for p in SOURCE_DIR.glob("*_classifier.py")}
    assert on_disk == set(METRIC_BRIDGES)
