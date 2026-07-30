"""Sonair CLI — image classification across training backends.

The framework is a first-class dimension of the command line::

    sonair lightning train    cfg.yaml     # implemented
    sonair lightning evaluate cfg.yaml
    sonair lightning predict  cfg.yaml
    sonair keras      train   cfg.yaml     # implemented (Keras 3)
    sonair keras      evaluate cfg.yaml
    sonair keras      predict  cfg.yaml
    sonair tensorflow train   cfg.yaml     # template — raises with instructions
    sonair jax        train   cfg.yaml     # template
    sonair huggingface train  cfg.yaml     # template
    sonair mlx        train   cfg.yaml     # template

What stays constant across backends is the point of the demo: the same
``runnable:`` + ``task:`` config, the same three verbs, and the same logging
backends. Only the runnable class named in the YAML changes. (A generic
``recordstream run cfg.yaml`` also works — it calls ``.run()`` with whatever
``task`` the YAML declares, with no framework verb at all.)

Adding a backend: write ``sonair/<name>_classifier.py`` (see
``docs/frameworks.md``), then replace its template block below with a real
sub-app modelled on ``lightning``.
"""

import sys
from typing import Any, Callable, List

from liquifai import LiquifaiError, LiquifyApp
from loggair import get_logger

logger = get_logger(__name__)

app = LiquifyApp(name="sonair", description="Image classification across training backends")


class BackendNotImplementedError(LiquifaiError, NotImplementedError):
    """A backend's verbs exist as a template but its runnable class does not.

    Dual-inherits ``LiquifaiError`` so liquifai renders one clean message +
    exit 1 (an unbuilt backend is a documented state, not a crash) and
    ``NotImplementedError`` so ordinary ``except`` clauses still catch it.
    """


def _dispatch(runnable: Any, task: str) -> None:
    """Build the bound runnable, force ``task``, and run it.

    The verb wins over any ``task:`` set in YAML, so one config serves all
    three verbs. Backend-agnostic: needs only a ``task`` attribute and ``run()``.

    Building goes through ``recordstream.cli.materialize_runnable`` — NOT a bare
    ``flow()`` — because the flat config's top-level keys (``train_set`` /
    ``batch_size`` / ``max_epochs`` / ``logger`` …) only broadcast into the
    constructor when the node is built against the document it came from. A bare
    flow drops them SILENTLY: `train_set` becomes None and `max_epochs` quietly
    falls back to the constructor default. Same helper as `recordstream run`, so
    both runners share one answer.
    """
    if runnable is None:
        raise ValueError("No runnable was bound — provide one under 'runnable:' in your YAML.")

    from recordstream.cli import materialize_runnable

    runnable = materialize_runnable(runnable)
    if hasattr(runnable, "task"):
        runnable.task = task  # the verb wins over the YAML's task
    runnable.run()


def _register_backend_classification(module: str) -> None:
    """Import a backend's loss/metric registrations so a config may NAME them.

    A ``!class:`` target that is a registered name (``CrossEntropyLoss``, or a selector
    like ``SparseCategoricalCrossentropy@role=loss``) resolves only once the module that
    registers it has been imported; a dotted path resolves through importlib either way.
    Discovery services bootstrap every ``confluid.configurables`` entry point, but a CLI
    does not — so each backend imports its OWN set, inside its own sub-app, where the
    framework is already the operator's choice.
    """
    import importlib

    importlib.import_module(module)


def _register_torch_classification() -> None:
    _register_backend_classification("sonair.torch_classification")


def _register_keras_classification() -> None:
    _register_backend_classification("sonair.keras_classification")


# --------------------------------------------------------------------------- #
# lightning — the implemented backend
# --------------------------------------------------------------------------- #

lightning = LiquifyApp(
    name="lightning",
    description="PyTorch Lightning — the reference backend (timm models, Lightning loggers).",
)


@lightning.script_command(flow_mode="manual")
def train(runnable: Any) -> None:
    """Train an image classifier with PyTorch Lightning (task=fit)."""
    _register_torch_classification()
    _dispatch(runnable, "fit")


@lightning.script_command(flow_mode="manual")
def evaluate(runnable: Any) -> None:
    """Evaluate a trained classifier over the test set (task=test)."""
    _register_torch_classification()
    _dispatch(runnable, "test")


@lightning.script_command(flow_mode="manual")
def predict(runnable: Any) -> None:
    """Stream predictions from a trained classifier to its sink (task=predict)."""
    _register_torch_classification()
    _dispatch(runnable, "predict")


app.add_app(lightning, "lightning")


# --------------------------------------------------------------------------- #
# keras — the second implemented backend
# --------------------------------------------------------------------------- #
# Written out plainly, like `lightning` above it and for the same reason: two sub-apps
# that each call `_dispatch` share three lines, which is less than any indirection that
# would remove the duplication. Revisit if a THIRD real backend lands.
#
# Keras 3 chooses its compute backend from `KERAS_BACKEND` at import time; sonair imports
# keras through `sonair._keras`, which defaults it to `torch` (the one this package
# guarantees) when the operator has not chosen. So `sonair keras train` works out of the
# box, and `KERAS_BACKEND=tensorflow sonair keras train` runs the identical config on
# TensorFlow once that is installed.

keras = LiquifyApp(
    name="keras",
    description="Keras 3 — one API over PyTorch / TensorFlow / JAX (set KERAS_BACKEND to pick).",
)


# `name=` is explicit because the handlers cannot both be called `train`: liquifai derives a
# command's name from the function name, and two module-level `def train` would collide.
@keras.script_command(name="train", flow_mode="manual")
def keras_train(runnable: Any) -> None:
    """Train an image classifier with Keras 3 (task=fit)."""
    _register_keras_classification()
    _dispatch(runnable, "fit")


@keras.script_command(name="evaluate", flow_mode="manual")
def keras_evaluate(runnable: Any) -> None:
    """Evaluate a trained classifier over the test set (task=test)."""
    _register_keras_classification()
    _dispatch(runnable, "test")


@keras.script_command(name="predict", flow_mode="manual")
def keras_predict(runnable: Any) -> None:
    """Stream predictions from a trained classifier to its sink (task=predict)."""
    _register_keras_classification()
    _dispatch(runnable, "predict")


app.add_app(keras, "keras")


# --------------------------------------------------------------------------- #
# Backend templates — verbs exist so `--help` and TAB advertise the roadmap,
# and each names the work instead of shrugging. Replace a block with a real
# sub-app (like `lightning` above) once its runnable class exists.
# --------------------------------------------------------------------------- #

TEMPLATES = {
    "tensorflow": "TensorFlow / tf.keras — graph execution and SavedModel export. TEMPLATE.",
    "jax": "JAX + Flax/NNX — functional transforms, jit/pmap scaling. TEMPLATE.",
    "huggingface": "Transformers `Trainer` + Accelerate — the fine-tuning path. TEMPLATE.",
    "mlx": "MLX — training on Apple-silicon unified memory. TEMPLATE.",
}

#: Every backend verb, implemented or not — used to map a verb to its scope.
BACKENDS = ("lightning", "keras", *TEMPLATES)


def _template_message(name: str, verb: str) -> str:
    """The actionable error a template verb raises — the file, the reference, the edit."""
    return (
        f"`sonair {name} {verb}` is not implemented yet — {name} is a backend TEMPLATE.\n\n"
        f"To implement it:\n"
        f"  1. Create sonair/{name}_classifier.py with a runnable class — a `task` Literal,\n"
        f"     `run()` -> `initialize()` -> `@entrypoint`-tagged methods, and\n"
        f'     `@configurable(task="classification", role="trainer")`.\n'
        f"     Reference: sonair/lightning_classifier.py (see docs/frameworks.md).\n"
        f"  2. Replace the `{name}` template in sonair/cli.py with a real sub-app,\n"
        f"     modelled on the `lightning` one above it.\n"
        f"  3. Install the backend: pip install 'sonair[{name}]'\n\n"
        f"The verbs, the config shape (`runnable:` + `task:`) and the logging backends\n"
        f"are already wired — only the runnable class is missing."
    )


def _template_command(name: str, verb: str) -> Callable[[], None]:
    """One unimplemented verb.

    A factory, not a default-argument closure: liquifai reads the handler's
    SIGNATURE, so smuggling the loop variables in as defaults would advertise
    them as ``--_name`` / ``--_verb`` options in help and TAB completion.

    The handler takes no ``runnable`` parameter on purpose — the failure must
    land before liquifai resolves a config for a class that does not exist, so
    the user gets instructions rather than an unresolvable-``!class:`` error.
    """

    def command() -> None:
        # No escaping here: liquifai >= 0.1.1 escapes the message in its own failure
        # renderer, so `pip install 'sonair[keras]'` survives with its extra intact.
        # (Against 0.1.0 it printed `pip install 'sonair'` — Rich read `[keras]` as a
        # style tag — and sonair escaped at the raise site to compensate.)
        raise BackendNotImplementedError(_template_message(name, verb))

    command.__name__ = f"{name}_{verb}"
    command.__doc__ = f"{verb.capitalize()} with {name} — NOT IMPLEMENTED (backend template)."
    return command


def _template_app(name: str, description: str) -> LiquifyApp:
    """Build a backend's sub-app with the three verbs, all unimplemented."""
    sub = LiquifyApp(name=name, description=description)
    for verb in ("train", "evaluate", "predict"):
        sub.command(name=verb)(_template_command(name, verb))
    return sub


for _name, _description in TEMPLATES.items():
    app.add_app(_template_app(_name, _description), _name)


def _inject_framework_scope(argv: List[str]) -> None:
    """Let the backend verb activate the matching ``framework=<name>`` scope.

    One config can carry every backend: shared keys at the top level (which is
    also the default backend) and a ``!scope:framework=<name>`` block per
    alternative, overriding only the framework-specific keys. Without this, a
    user would name the backend twice —
    ``sonair keras train cfg.yaml --scope framework=keras``.

    Appends to ``argv`` rather than reaching into liquifai: scopes are parsed
    from the command line in phase 3, so this must land before ``run()``. An
    explicit ``--scope framework=…`` or ``--framework …`` always wins.
    """
    for i, arg in enumerate(argv):
        if arg.startswith("--framework"):  # `--framework keras` / `--framework=keras`
            return
        if arg.startswith(("--scope=", "-s=")) and "framework=" in arg:
            return
        if arg in ("--scope", "-s") and i + 1 < len(argv) and argv[i + 1].startswith("framework="):
            return
    backend = next((a for a in argv[1:] if a in BACKENDS), None)
    if backend is not None:
        argv += ["--scope", f"framework={backend}"]


def main() -> None:
    _inject_framework_scope(sys.argv)
    app.run()


if __name__ == "__main__":
    main()
