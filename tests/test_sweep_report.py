"""``examples/sweep/sweep_report.py`` — how a sweep is read back and rendered.

The script is an EXAMPLE and not part of the package, so it is loaded by path. It is tested
anyway for the reason the workspace tests examples at all: it is the surface a reader meets
after a thirteen-run sweep, and a report that quietly shows the wrong columns is worse than no
report — the numbers look authoritative either way.

What is pinned here is the RENDERING, which is pure. The two tracker-reading halves need a live
MLflow or a written-out CSV tree and are exercised by running the sweep.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, List

import pytest

_REPORT_PATH = Path(__file__).resolve().parent.parent / "examples" / "sweep" / "sweep_report.py"


def _load_report_module() -> Any:
    spec = importlib.util.spec_from_file_location("sweep_report", _REPORT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


report = _load_report_module()


def _rows() -> List[Dict[str, Any]]:
    """Two runs shaped like the real thing: one torch backend, one `transformers.Trainer`.

    The keys are what a real sweep recorded (`sweep_logs/csv/<experiment>/<run>/metrics.csv`)
    after the eight backends were aligned on one vocabulary.
    """
    return [
        {
            "run": "huggingface_zoo",
            "status": "FINISHED",
            "seconds": 12.0,
            "metrics": {
                "train/loss": 1.31,
                "val/loss": 1.37,
                "val/accuracy": 0.65,
                "epoch": 1.0,
                "train/runtime": 3.01,
                "train/samples_per_second": 331.9,
                "train/total_flos": 1.0e16,
                "val/runtime": 0.19,
            },
        },
        {
            "run": "torch_zoo",
            "status": "FINISHED",
            "seconds": 71.0,
            "metrics": {
                "train/loss": 1.11,
                # Lightning's per-BATCH series of the SAME quantity — a real result, and the one
                # column a thirteen-run table would leave twelve-thirteenths blank.
                "train/loss_step": 1.09,
                "val/loss": 1.81,
                "val/accuracy": 0.43,
                "val/f1_macro": 0.38,
            },
        },
    ]


def test_bookkeeping_columns_are_folded_away_by_default() -> None:
    """A `transformers.Trainer` records its timings beside the loss; a comparison table is not
    where a reader wants them, and thirteen runs of mostly-blank columns hide the three that
    every backend fills."""
    header = report.render(_rows(), "exp", "csv", None).splitlines()[0]

    assert "val/accuracy" in header and "val/loss" in header and "train/loss" in header
    assert "train/total_flos" not in header
    assert "train/samples_per_second" not in header


def test_what_was_hidden_is_named_with_the_command_that_shows_it() -> None:
    """Never a silent trim: the footer says how many columns went, which, and how to get them."""
    table = report.render(_rows(), "Tue_Aug_4_2026_1", "csv", None)

    assert "columns hidden — bookkeeping" in table
    assert "train/total_flos" in table  # named in the footer even though its column is gone
    assert "python examples/sweep/sweep_report.py report csv Tue_Aug_4_2026_1 --all" in table


def test_show_all_puts_every_recorded_value_back() -> None:
    table = report.render(_rows(), "exp", "csv", None, show_all=True)

    assert "train/total_flos" in table and "epoch" in table
    assert "hidden" not in table


def test_a_metric_only_one_backend_computes_keeps_its_column() -> None:
    """The union, not a hand-picked list: `val/f1_macro` is a real result one engine reports."""
    assert "val/f1_macro" in report.render(_rows(), "exp", "csv", None)


def test_validation_columns_come_first() -> None:
    """That is the column a sweep is read for — the training-split ones are context."""
    table = report.render(_rows(), "exp", "csv", None)
    header = table.splitlines()[0]

    assert header.index("val/accuracy") < header.index("train/loss")


@pytest.mark.parametrize(
    "key,bookkeeping",
    [
        ("epoch", True),
        ("train/runtime", True),
        ("val/samples_per_second", True),
        ("train/total_flos", True),
        ("train/loss", False),
        ("val/accuracy", False),
        ("val/top_k_accuracy", False),
    ],
)
def test_the_rule_reads_the_BASE_name(key: str, bookkeeping: bool) -> None:
    """Split-agnostic: `train/runtime` and `val/runtime` are the same kind of column."""
    assert report.is_bookkeeping(key) is bookkeeping


def test_a_tracker_that_cannot_be_read_says_where_to_look_instead() -> None:
    """Unchanged by the trimming, and the reason `render` takes a `note` at all: an empty table
    from a write-only backend reads as "the sweep recorded nothing"."""
    rows, note = report.collect("wandb", "exp")

    assert rows == [] and note is not None
    assert report.render(rows, "exp", "wandb", note).strip() == note


def test_a_within_epoch_series_is_folded_but_is_NOT_called_bookkeeping() -> None:
    """`train/loss_step` is a genuine result, so it gets its own reason and its own word.

    Lightning appends `_step` to a name it logs per batch as well as per epoch. That series is
    worth recording — nothing else shows the loss moving WITHIN an epoch, and it drives the
    progress bar — but a one-row-per-run table cannot show a finer axis, and only the backend
    whose engine emits the suffix fills it. Folding it under `BOOKKEEPING` would have been two
    fewer lines and would have made the footer call a metric something it is not.
    """
    assert report.is_within_epoch_series("train/loss_step")
    assert not report.is_bookkeeping("train/loss_step")
    assert report.is_folded("train/loss_step")

    table = report.render(_rows(), "exp", "csv", None)
    header = table.splitlines()[0]
    assert "train/loss" in header
    assert "train/loss_step" not in header
    assert "within-epoch series" in table and "train/loss_step" in table  # named in the footer


def test_the_footer_names_only_the_reasons_that_apply() -> None:
    """A run with no timings should not be told its bookkeeping was hidden."""
    rows = [{"run": "a", "status": "FINISHED", "seconds": 1.0, "metrics": {"train/loss": 1.0, "train/loss_step": 1.1}}]
    table = report.render(rows, "exp", "csv", None)

    assert "within-epoch series" in table
    assert "bookkeeping" not in table


def test_a_bare_step_suffix_is_not_mistaken_for_a_series() -> None:
    """`step` is an AXIS key (already bookkeeping); the rule needs a base name in front of it."""
    assert not report.is_within_epoch_series("step")
    assert not report.is_within_epoch_series("train/_step")
    assert report.is_bookkeeping("step")


def test_show_all_puts_the_within_epoch_series_back_too() -> None:
    assert "train/loss_step" in report.render(_rows(), "exp", "csv", None, show_all=True).splitlines()[0]
