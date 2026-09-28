"""Every example config must LOAD + FLOW to a runnable — the CI safety net.

CI runs ``python examples/*.py`` (an aisland-scaffolded glob), but sonair ships
its examples as ``.yaml`` configs for ``recordstream run``, which the glob never
executes. Without this test a stale example (a renamed class, a bad ``!class:``
path, a wrong ``task:``) would fail only at run time and sail through CI green.

This test mirrors exactly what ``recordstream run`` does short of training: it
loads each ``examples/*.yaml``, resolves + flows the object bound under the
top-level ``runnable:`` key (importing its ``!class:``/``!lazy:`` targets and
constructing it — cheap, since every configurable is zero-arg / lazy per the
workspace lazy-construction mandate, so no dataset download or training runs),
and asserts it exposes a callable ``run()`` and a valid ``task``. A dangling
class path raises ``confluid.UnknownClassError`` at flow time, failing the test.
"""

from pathlib import Path

import confluid
import pytest
from confluid import flow
from confluid.fluid import Fluid

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
EXAMPLE_CONFIGS = sorted(EXAMPLES_DIR.glob("*.yaml"))
VALID_TASKS = {"fit", "evaluate", "test", "predict"}


def test_examples_dir_is_not_empty() -> None:
    """Guard: if the examples move/rename, this test must not silently cover nothing."""
    assert EXAMPLE_CONFIGS, f"no example configs found under {EXAMPLES_DIR}"


@pytest.mark.parametrize("config_path", EXAMPLE_CONFIGS, ids=lambda p: p.name)
def test_example_config_loads_and_flows_to_a_runnable(config_path: Path) -> None:
    """The config binds a ``runnable:`` whose ``!class:`` resolves + constructs, with a valid task."""
    document = confluid.load(config_path.read_text())
    runnable = document["runnable"] if isinstance(document, dict) else getattr(document, "runnable", None)
    assert runnable is not None, f"{config_path.name}: no top-level 'runnable:' key"

    if isinstance(runnable, Fluid):
        runnable = flow(runnable)  # imports the !class:/!lazy: target — dangling paths raise here

    assert callable(getattr(runnable, "run", None)), f"{config_path.name}: runnable has no run() method"
    task = getattr(runnable, "task", None)
    assert task in VALID_TASKS, f"{config_path.name}: invalid task {task!r} (expected one of {VALID_TASKS})"
