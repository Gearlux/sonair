"""The framework axis of the CLI — `sonair <verb> cfg.yaml --framework <name>`.

The backend is a dimension of the CONFIG that the CLI passes through, not a
subcommand. These tests pin the seams that make that honest:

1. Three verbs, each binding the config's ``runnable:`` key and forcing the
   verb's task (the verb wins over the YAML).
2. The registrations pre-step imports THIS project's modules (the machinery —
   the ``--framework`` rewrite, the keras engine pin, the dispatch guard, their
   ORDER — is `matrainer.cli`'s since 2026-08-07 and is pinned in matrainer's
   own `tests/test_cli.py`; what stays here is sonair's half of each seam).
3. The shipped example resolves to a complete runnable under EVERY framework it
   declares. That is the regression pin for the bug this shape was built to
   remove: a scope block swapping ``runnable:`` used to take the ``model:``
   nested under it, and ``--framework torch`` died with
   ``TorchClassifier.__init__() missing 1 required positional argument: 'model'``.
"""

import inspect
import tomllib
from pathlib import Path
from typing import Any, Dict, List, Set

import confluid
import pytest
from confluid import configurable
from matrainer.cli import VERB_TASKS, Framework, default_dispatch, register_frameworks, registration_modules

from sonair import cli
from sonair.cli import FRAMEWORKS

VERBS = ("train", "evaluate", "predict")

#: The backends sonair ships, restated here rather than imported blind — this list is the
#: test's own claim, so one silently disappearing from `cli.FRAMEWORKS` has to fail something.
EXPECTED_FRAMEWORKS = ("lightning", "torch", "keras", "tensorflow", "fastai", "huggingface", "jax", "mlx")

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "train_efficientvit_b0_mnist.yaml"


def test_the_framework_set_is_the_one_we_expect() -> None:
    assert tuple(FRAMEWORKS) == EXPECTED_FRAMEWORKS


def test_the_verbs_are_the_whole_command_surface() -> None:
    """Three verbs plus the listing — the backend is a flag, not 24 subcommands."""
    assert set(cli.app._commands) == {*VERBS, "frameworks"}
    assert not cli.app._sub_apps, "a backend must not reappear as a sub-app"


@pytest.mark.parametrize("verb", VERBS)
def test_every_verb_takes_a_promoted_config_and_a_runnable(verb: str) -> None:
    """`@script_command` gives config promotion; the param NAME drives liquifai DI.

    The handlers are built by a factory, and the one thing a factory can get wrong is the
    signature — liquifai reads it, so a smuggled loop variable would surface as a `--_verb`
    flag in help and TAB completion.
    """
    assert verb in cli.app._script_cmds
    assert list(inspect.signature(cli.app._commands[verb]).parameters) == ["runnable"]


def test_the_frameworks_listing_names_every_backend(capsys: Any) -> None:
    """`sonair frameworks` replaces what the eight sub-apps used to advertise in --help."""
    cli.app._commands["frameworks"]()
    out = capsys.readouterr().out
    for name, spec in FRAMEWORKS.items():
        assert name in out
        assert spec.description in out


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_every_framework_names_a_real_registration_module(framework: str) -> None:
    """The loss/metric module must exist — a typo would surface only at run time, as an
    unresolvable `!class:` name several minutes into a session.

    `None` is a legitimate answer, not a missing one: a backend whose engine computes its own
    loss (huggingface) has nothing a config could wire, so there is no module to name.
    """
    import importlib.util

    for module in registration_modules(FRAMEWORKS[framework]):
        assert importlib.util.find_spec(module) is not None, f"{framework} names a missing module {module!r}"


def test_tensorflow_shares_the_keras_registrations() -> None:
    """In TF >= 2.16 `tf.keras` IS Keras 3, so a separate `tensorflow_classification.py` would
    register the same classes under a second framework tag — making every bare
    `!class:SparseCategoricalCrossentropy` ambiguous for no gain."""
    assert FRAMEWORKS["tensorflow"].registrations == FRAMEWORKS["keras"].registrations


def test_every_framework_names_a_declared_pip_extra() -> None:
    """`sonair frameworks` reads as an install menu, so each name must be installable."""
    extras = tomllib.loads((EXAMPLE.parent.parent / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    for framework in FRAMEWORKS:
        assert framework in extras, f"--framework {framework} has no matching `sonair[{framework}]` extra"


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
    cli.app._commands[verb](runnable=runnable)
    assert runnable.task == task, "the verb must win over the YAML's task"
    assert runnable.ran


@configurable
class _ConfigRunnable:
    """A runnable with the ergonomic-knob shape a flat sonair config drives."""

    def __init__(self, task: str = "fit", max_epochs: int = 10, experiment_name: str = "default") -> None:
        self.task, self.max_epochs, self.experiment_name = task, max_epochs, experiment_name
        self.ran = False

    def run(self) -> None:
        self.ran = True


def test_dispatch_builds_with_the_flat_configs_top_level_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """The verbs must NOT bare-flow the bound node — that drops every top-level key.

    A bare ``flow()`` leaves ``train_set`` at ``None`` and ``max_epochs`` at the
    constructor default, SILENTLY, and the run proceeds looking configured.
    """
    from liquifai.context import LiquifyContext, get_context, set_context

    text = (
        "runnable: !class:tests.test_cli_frameworks._ConfigRunnable\n"
        "max_epochs: 3\n"
        "experiment_name: from_the_flat_config\n"
    )
    document = confluid.load(text, until="document")
    previous = get_context()
    ctx = LiquifyContext(name="test")
    ctx.config_data = document
    set_context(ctx)
    try:
        built: dict = {}
        monkeypatch.setattr(_ConfigRunnable, "run", lambda self: built.update(vars(self)))
        default_dispatch(document["runnable"], "predict")
    finally:
        set_context(previous)

    assert built["max_epochs"] == 3, "a bare flow() leaves this at the ctor default of 10 — silently"
    assert built["experiment_name"] == "from_the_flat_config"
    assert built["task"] == "predict", "the verb still wins over the config"


# ---------------------------------------------------------------------------
# The registrations pre-step, against THIS project's table
# ---------------------------------------------------------------------------


def test_registration_imports_the_frameworks_own_module(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    seen: List[str] = []
    monkeypatch.setattr(importlib, "import_module", lambda name: seen.append(name))
    register_frameworks("keras", FRAMEWORKS)

    assert seen == ["sonair.keras_classification"]


@pytest.mark.parametrize("framework", [None, "huggingface", "not-a-framework"])
def test_registration_imports_nothing_when_there_is_nothing_to_import(
    monkeypatch: pytest.MonkeyPatch, framework: Any
) -> None:
    """Three real states, none an error: no flag (the engine is unnamed), a backend whose
    model computes its own loss, and a name that is not a backend at all."""
    import importlib

    seen: List[str] = []
    monkeypatch.setattr(importlib, "import_module", lambda name: seen.append(name))
    register_frameworks(framework, FRAMEWORKS)

    assert seen == []


# ---------------------------------------------------------------------------
# The shipped example resolves under EVERY framework it declares
# ---------------------------------------------------------------------------


def _declared_frameworks() -> Set[str]:
    return confluid.discover_dimension_values(confluid.load(EXAMPLE, until="raw")).get("framework", set())


def test_the_example_declares_its_frameworks_as_scope_values() -> None:
    """Including the DEFAULT one: every backend is a `!scope:framework=<name>` block,
    `default_scopes:` names the one a bare run uses, and no top-level `runnable:` is
    privileged — confluid rejects an activation that matches no block, and the
    default goes through the same check."""
    declared = _declared_frameworks()
    assert "lightning" in declared, "the default backend must be nameable, not only implicit"
    assert declared <= set(FRAMEWORKS), f"the config declares a backend the CLI does not know: {declared}"
    raw = confluid.load(EXAMPLE, until="raw")
    assert raw["default_scopes"] == ["framework=lightning"]
    assert "runnable" not in raw, "the runnable lives in each framework block, not at the top level"
    assert confluid.load(EXAMPLE, until="document")["runnable"].target == "sonair.LightningClassifier"


def test_every_declared_framework_resolves_to_a_complete_runnable() -> None:
    """The regression pin. A framework block swaps `runnable:`, and a key NESTED under
    `runnable:` dies with it — which is exactly how `model:` used to vanish for the
    torch and fastai backends while their comments claimed it was inherited.

    Config-level on purpose: no framework has to be installed for this to run, so it
    fails on the machine that introduces the mistake rather than only where the extra
    happens to be present.
    """
    for framework in sorted(_declared_frameworks()):
        document = confluid.load(EXAMPLE, until="document", scopes=[f"framework={framework}"])
        assert isinstance(document, dict)
        runnable = document.get("runnable")
        assert runnable is not None, f"--framework {framework}: no runnable"
        model = runnable.kwargs.get("model") if hasattr(runnable, "kwargs") else None
        assert model is not None or document.get("model") is not None, (
            f"--framework {framework}: no model reachable — a block swapped `runnable:` and took "
            f"a nested `model:` with it. Keep every framework-specific key a top-level sibling."
        )


def test_no_framework_inherits_a_metric_set_its_engine_cannot_consume() -> None:
    """`metrics:` is ENGINE-BOUND, so an inherited top-level value is only right for some blocks.

    The example's top-level set is torchmetrics (`framework="torch"`), which the three
    torch-backed backends read unchanged — but `keras.Model.compile()` cannot take a
    `MulticlassAccuracy`, so a Keras-engine block MUST override the key (`metrics: null`
    puts it back to the classifier's own sentinel). Config-level for the same reason as the
    test above: it fails wherever the mistake is made, not only where keras is installed.
    """
    torch_backed = {"lightning", "torch", "fastai"}
    for framework in sorted(_declared_frameworks()):
        document = confluid.load(EXAMPLE, until="document", scopes=[f"framework={framework}"])
        assert isinstance(document, dict)
        for key in ("metrics", "val_metrics"):
            value = document.get(key)
            if framework in torch_backed:
                assert value, f"--framework {framework}: runs on torch, so it should inherit `{key}`"
            else:
                assert not value, (
                    f"--framework {framework}: inherited the top-level torchmetrics `{key}`, which its "
                    f"engine cannot consume. Override it in that block (`{key}: null` restores the sentinel)."
                )
        if framework in torch_backed:
            assert "confusion" in document["val_metrics"], "the confusion matrix belongs to the validation split"
            assert "confusion" not in document["metrics"], "...and must NOT be in the shared, every-split set"


def test_the_example_keeps_the_engine_agnostic_keys_shared() -> None:
    """One config for every backend: shared top level + scoped overrides."""
    default = confluid.load(EXAMPLE, until="document")
    assert isinstance(default, dict)
    assert "runnable" in default, "no scope -> the default backend must still be complete"
    for shared in ("train_set", "val_set", "batch_size", "max_epochs", "experiment_name", "preprocess"):
        assert shared in default, f"{shared} must be shared, not framework-specific"


def test_an_undeclared_framework_is_rejected_with_the_real_values() -> None:
    """The check the backend verb could never do — it named a backend nothing compared
    against the config, so a mismatch trained the default and said nothing."""
    with pytest.raises(confluid.ScopeError) as ei:
        confluid.load(EXAMPLE, until="document", scopes=["framework=kears"])

    assert "keras" in str(ei.value), "the error must list the backends the config DOES declare"


def test_framework_is_a_two_field_record_of_data() -> None:
    """No per-backend BEHAVIOUR in the table: the keras engine pin would be a third
    column that is None seven times out of eight, which is what keeps it in the
    shared machinery (`matrainer.cli.pin_keras_backend`)."""
    assert Framework._fields == ("description", "registrations")
    assert all(isinstance(spec, Framework) for spec in FRAMEWORKS.values())


def test_the_frameworks_table_carries_no_duplicate_descriptions() -> None:
    """`sonair frameworks` is the discovery surface — two backends reading alike is a
    copy-paste that leaves one of them undescribed."""
    seen: Dict[str, str] = {}
    for name, spec in FRAMEWORKS.items():
        assert spec.description not in seen, f"{name} repeats {seen[spec.description]}'s description"
        seen[spec.description] = name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
