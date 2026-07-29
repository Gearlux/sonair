"""The framework axis of the CLI — `sonair <backend> <verb>`.

Pins three things:

1. Every backend is mounted with the same three verbs, so a template advertises
   the roadmap instead of hiding it.
2. An IMPLEMENTED backend's verb binds the config's ``runnable:`` key and forces
   the verb's task (the verb wins over the YAML).
3. A TEMPLATE backend's verb fails through liquifai's failure contract — one
   clean message + exit 1, naming the work — not an unhandled crash.
"""

import inspect
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest

from sonair import cli
from sonair.cli import TEMPLATES, BackendNotImplementedError

VERBS = ("train", "evaluate", "predict")
VERB_TASKS = {"train": "fit", "evaluate": "test", "predict": "predict"}


def test_lightning_and_the_templates_are_all_mounted() -> None:
    assert set(cli.app._sub_apps) == {"lightning", *TEMPLATES}


@pytest.mark.parametrize("backend", ["lightning", *TEMPLATES])
def test_every_backend_exposes_the_same_verbs(backend: str) -> None:
    """Templates advertise train/evaluate/predict too — the roadmap is visible."""
    assert set(cli.app._sub_apps[backend]._commands) == set(VERBS)


@pytest.mark.parametrize("verb", VERBS)
def test_lightning_verbs_take_a_promoted_config_and_a_runnable(verb: str) -> None:
    """`@script_command` gives config promotion; the param name drives liquifai DI."""
    sub = cli.app._sub_apps["lightning"]
    assert verb in sub._script_cmds
    assert list(inspect.signature(sub._commands[verb]).parameters) == ["runnable"]


@pytest.mark.parametrize("backend", TEMPLATES)
@pytest.mark.parametrize("verb", VERBS)
def test_template_verbs_take_no_config(backend: str, verb: str) -> None:
    """A template must fail before liquifai resolves a config for a missing class."""
    sub = cli.app._sub_apps[backend]
    assert verb not in sub._script_cmds
    assert list(inspect.signature(sub._commands[verb]).parameters) == []


# ---------------------------------------------------------------------------
# Dispatch: the verb wins over the YAML's task
# ---------------------------------------------------------------------------


class _FakeRunnable:
    def __init__(self, task: str = "fit") -> None:
        self.task = task
        self.ran = False

    def run(self) -> None:
        self.ran = True


@pytest.mark.parametrize("verb,task", sorted(VERB_TASKS.items()))
def test_verb_overrides_the_configured_task(verb: str, task: str) -> None:
    runnable = _FakeRunnable(task="something-else")
    cli.app._sub_apps["lightning"]._commands[verb](runnable=runnable)
    assert runnable.task == task, "the verb must win over the YAML's task"
    assert runnable.ran


def test_dispatch_without_a_runnable_is_a_clear_error() -> None:
    with pytest.raises(ValueError, match="runnable:"):
        cli._dispatch(None, "fit")


# ---------------------------------------------------------------------------
# Templates fail informatively
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", TEMPLATES)
def test_template_verb_raises_with_instructions(backend: str) -> None:
    with pytest.raises(BackendNotImplementedError) as ei:
        cli.app._sub_apps[backend]._commands["train"]()

    message = str(ei.value)
    assert "not implemented yet" in message
    assert f"sonair/{backend}_classifier.py" in message  # the file to create
    assert "lightning_classifier.py" in message  # the reference to copy
    assert "sonair/cli.py" in message  # where to replace the template
    # The pip extra survives Rich's markup parser (an unescaped `[keras]` would
    # be eaten, printing `pip install 'sonair'` — a wrong instruction).
    assert f"sonair\\[{backend}]" in message


@pytest.mark.parametrize("backend", TEMPLATES)
def test_template_error_is_a_notimplementederror_too(backend: str) -> None:
    """Dual inheritance: liquifai renders it cleanly, `except NotImplementedError` still catches."""
    with pytest.raises(NotImplementedError):
        cli.app._sub_apps[backend]._commands["predict"]()


def test_template_exits_1_through_the_cli(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    """End-to-end: one clean `Error:` line, exit 1 — not an unhandled traceback."""
    from liquifai.context import set_context

    monkeypatch.setattr(sys, "argv", ["sonair", "keras", "train"])
    set_context(None)  # type: ignore[arg-type]

    with pytest.raises(SystemExit) as ei:
        cli.app.run()

    assert ei.value.code == 1
    out = capsys.readouterr().out
    assert "Error:" in out
    assert "not implemented yet" in out


def test_every_template_names_a_declared_pip_extra() -> None:
    """The install hint must name a real extra, or it lies."""
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    extras = tomllib.loads(pyproject.read_text())["project"]["optional-dependencies"]
    for backend in TEMPLATES:
        assert backend in extras, f"{backend} hint says `sonair[{backend}]` but pyproject has no such extra"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ---------------------------------------------------------------------------
# The backend verb activates the matching `framework=<name>` scope
# ---------------------------------------------------------------------------
# One config carries every backend: shared keys at the top level (also the
# default backend) and a `!scope:framework=<name>` block per alternative. The
# verb picks the scope so the backend is never named twice.


@pytest.mark.parametrize("backend", ["lightning", *TEMPLATES])
def test_verb_injects_its_framework_scope(backend: str) -> None:
    argv = ["sonair", backend, "train", "cfg.yaml"]
    cli._inject_framework_scope(argv)
    assert argv[-2:] == ["--scope", f"framework={backend}"]


def test_no_backend_verb_injects_nothing() -> None:
    """`sonair --help` and unknown tokens must not gain a scope."""
    argv = ["sonair", "--help"]
    cli._inject_framework_scope(argv)
    assert argv == ["sonair", "--help"]


@pytest.mark.parametrize(
    "explicit",
    [
        ["--scope", "framework=keras"],
        ["--scope=framework=keras"],
        ["-s", "framework=keras"],
        ["--framework", "keras"],
        ["--framework=keras"],
    ],
)
def test_an_explicit_framework_choice_wins(explicit: list) -> None:
    """Never override what the user typed, even if the verb says otherwise."""
    argv = ["sonair", "lightning", "train", "cfg.yaml", *explicit]
    before = list(argv)
    cli._inject_framework_scope(argv)
    assert argv == before


def test_shipped_example_separates_shared_from_framework_specific_keys() -> None:
    """The example is one config for every backend: shared top level + scoped overrides.

    Every alternative backend is still commented out (their runnable classes do
    not exist), so the file declares NO `framework` dimension yet. This pins the
    half that is real today: the default backend is complete on its own, and the
    engine-agnostic keys sit at the top level where a scope block never has to
    repeat them.
    """
    import confluid

    path = Path(__file__).resolve().parent.parent / "examples" / "train_efficientvit_b0_mnist.yaml"
    raw = confluid.load_config(path)

    default = confluid.load(raw, flow=False)
    assert "runnable" in default, "no scope -> the default backend must still be complete"
    for shared in ("train_set", "val_set", "batch_size", "max_epochs", "experiment_name"):
        assert shared in default, f"{shared} must be shared, not framework-specific"


def test_injected_scope_is_inert_before_any_alternative_exists() -> None:
    """`sonair lightning train` injects `framework=lightning` unconditionally.

    While the alternatives are comments the config references no such scope, so
    the injected value must be an inert no-op rather than an error or a dropped key.
    """
    import confluid

    path = Path(__file__).resolve().parent.parent / "examples" / "train_efficientvit_b0_mnist.yaml"
    raw = confluid.load_config(path)
    assert (
        confluid.load(raw, flow=False, scopes=["framework=lightning"]).keys() == confluid.load(raw, flow=False).keys()
    )
