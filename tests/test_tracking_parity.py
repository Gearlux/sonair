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
def test_every_backend_hands_marainer_the_runnable(stem: str) -> None:
    """`runnable=self`, never a params dict the backend transcribes for itself."""
    entries = calls_with_runnable_self(module_source(stem))
    assert entries, (
        f"{stem} does not pass `runnable=self` to any tracking entry point. "
        "Hand marainer the object so the hyperparameters come from its one derivation "
        "(provenance.config_params), not from a dict this backend writes by hand."
    )
    assert entries <= {"log_run_context", "TrackingCallback", "keras_tracking_callback"}, entries


@pytest.mark.parametrize("stem", BACKENDS)
def test_every_backend_captures_run_provenance(stem: str) -> None:
    """Wiring `trackers:` must record the run, not just its class names."""
    names = called_names(module_source(stem))
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
    """
    source = module_source(stem)
    names = called_names(source)
    if names & {"TrackingCallback", "keras_tracking_callback"}:
        return  # the callback owns the lifecycle
    assert "capture_log_file" in names, (
        f"{stem} starts a manifest via log_run_context but never finishes it; call "
        "capture_log_file(trackers, manifest=..., status='success') at the end of fit()."
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
    loop calls `log_metrics` itself, a marainer callback forwards `on_epoch_end`, or the
    framework's own history is replayed after the fit.
    """
    source = module_source(stem)
    names = called_names(source)
    forwards = "log_metrics" in names
    via_callback = bool(names & {"TrackingCallback", "keras_tracking_callback"})
    replays = bool(names & {"_log_recorded_epochs", "_log_trainer_history"})
    assert forwards or via_callback or replays, (
        f"{stem} never forwards metrics to its trackers — a wired backend would record the "
        "config and draw an empty curve."
    )
