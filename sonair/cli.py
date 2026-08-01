"""Sonair CLI — image classification across training backends.

The framework is a first-class dimension of the command line::

    sonair lightning train    cfg.yaml     # implemented
    sonair lightning evaluate cfg.yaml
    sonair lightning predict  cfg.yaml
    sonair keras      train   cfg.yaml     # implemented (Keras 3)
    sonair tensorflow train   cfg.yaml     # implemented (tf.data + GradientTape)
    sonair fastai     train   cfg.yaml     # implemented (Learner + one-cycle)
    sonair jax        train   cfg.yaml     # template — raises with instructions
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

import os
import sys
from typing import Any, Callable, Dict, List, NamedTuple, Optional

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


# --------------------------------------------------------------------------- #
# The implemented backends
# --------------------------------------------------------------------------- #
# Until 2026-07-30 `lightning` and `keras` were written out plainly — six decorated
# functions — with the recorded rule "if a THIRD real backend lands, factor it out THEN,
# with three concrete cases to generalise from" (docs/architecture.md §1). `tensorflow`
# and `fastai` are the third and fourth, and the four sub-apps differ in exactly three
# values: the name, the description, and which loss/metric module to import. So the
# generalisation is a table of those three values plus a factory, and nothing else —
# no `Framework` dataclass, no registry module, no per-backend subclass.
#
# What did NOT get folded in is the pre-import in `main()` below: `tensorflow` has to pin
# KERAS_BACKEND before anything imports keras, which is earlier than any handler can run.
# That asymmetry is real, so it stays visible instead of being hidden behind the table.


class Backend(NamedTuple):
    """One implemented backend: what to call it, what to say about it, what to import."""

    description: str
    #: The module whose import registers this engine's losses/metrics under a plain NAME.
    #: `tensorflow` shares the Keras set on purpose — in TF >= 2.16 `tf.keras` IS Keras 3,
    #: so a separate `tensorflow_classification.py` would register the same classes twice
    #: and make every bare `!class:SparseCategoricalCrossentropy` ambiguous for no gain.
    #:
    #: `None` means the backend has NOTHING to register, which is a real state rather than an
    #: oversight: a `transformers` model computes its own loss from `config.problem_type`, so
    #: there is no loss a config could wire (`docs/architecture.md` §5).
    registrations: Optional[str] = None


IMPLEMENTED: Dict[str, Backend] = {
    "lightning": Backend(
        "PyTorch Lightning — the reference backend (timm models, Lightning loggers).",
        "sonair.torch_classification",
    ),
    # Shares `lightning`'s loss/metric set, and that is the point rather than an economy: a
    # `torch.nn` loss does not care whether a Trainer or a hand-written loop calls it, so the
    # two backends differ ONLY in the loop.
    "torch": Backend(
        "PyTorch — a hand-written loop, no training framework at all (the control experiment).",
        "sonair.torch_classification",
    ),
    "keras": Backend(
        "Keras 3 — one API over PyTorch / TensorFlow / JAX (set KERAS_BACKEND to pick).",
        "sonair.keras_classification",
    ),
    "tensorflow": Backend(
        "TensorFlow — a native tf.data + tf.GradientTape loop, with SavedModel export.",
        "sonair.keras_classification",
    ),
    "fastai": Backend(
        "fastai — a Learner with one-cycle scheduling, over the same torch stack.",
        "sonair.fastai_classification",
    ),
    "huggingface": Backend(
        "Transformers `Trainer` + Accelerate — fine-tuning a Hub checkpoint.",
        # No registrations: the model computes its own loss (see the Backend field note).
    ),
    "jax": Backend(
        "JAX — a jitted optax loop over a Flax model (functional transforms).",
        # No registrations: optax losses are plain FUNCTIONS, so a config names them.
    ),
    "mlx": Backend(
        "MLX — training on Apple-silicon unified memory (macOS arm64 only).",
        # No registrations: `mlx.nn.losses` are plain FUNCTIONS, so a config names them.
    ),
}

#: Verb -> the ``task`` it forces. The verb wins over any ``task:`` in the YAML.
VERB_TASKS = {"train": "fit", "evaluate": "test", "predict": "predict"}


def _backend_command(verb: str, backend: str, spec: Backend) -> Callable[[Any], None]:
    """One implemented verb, bound to its backend.

    A factory, not a default-argument closure: liquifai reads the handler's SIGNATURE to
    build the command, so smuggling the loop variables in as defaults would advertise them
    as ``--_backend`` / ``--_verb`` options in help and TAB completion. The signature it
    must expose is exactly ``(runnable)`` — the parameter NAME is what liquifai's DI binds
    to the config's ``runnable:`` key.
    """

    def command(runnable: Any) -> None:
        if spec.registrations is not None:
            _register_backend_classification(spec.registrations)
        _dispatch(runnable, VERB_TASKS[verb])

    command.__name__ = f"{backend}_{verb}"
    command.__doc__ = f"{_VERB_DOCS[verb]} ({backend}; task={VERB_TASKS[verb]})."
    return command


_VERB_DOCS = {
    "train": "Train an image classifier",
    "evaluate": "Evaluate a trained classifier over the test set",
    "predict": "Stream predictions from a trained classifier to its sink",
}


def _backend_app(backend: str, spec: Backend) -> LiquifyApp:
    """Build an implemented backend's sub-app: the same three verbs, all real."""
    sub = LiquifyApp(name=backend, description=spec.description)
    for verb in VERB_TASKS:
        sub.script_command(name=verb, flow_mode="manual")(_backend_command(verb, backend, spec))
    return sub


for _backend, _spec in IMPLEMENTED.items():
    app.add_app(_backend_app(_backend, _spec), _backend)


# --------------------------------------------------------------------------- #
# Backend templates — verbs exist so `--help` and TAB advertise the roadmap,
# and each names the work instead of shrugging. Graduating one means writing its
# runnable, deleting its entry here, and adding it to IMPLEMENTED above.
# --------------------------------------------------------------------------- #

TEMPLATES: Dict[str, str] = {}

#: Every backend verb, implemented or not — used to map a verb to its scope.
BACKENDS = (*IMPLEMENTED, *TEMPLATES)


def _template_message(name: str, verb: str) -> str:
    """The actionable error a template verb raises — the file, the reference, the edit."""
    return (
        f"`sonair {name} {verb}` is not implemented yet — {name} is a backend TEMPLATE.\n\n"
        f"To implement it:\n"
        f"  1. Create sonair/{name}_classifier.py with a runnable class — a `task` Literal,\n"
        f"     `run()` -> `initialize()` -> `@entrypoint`-tagged methods, and\n"
        f'     `@configurable(task="classification", role="trainer")`.\n'
        f"     Reference: sonair/lightning_classifier.py (see docs/frameworks.md).\n"
        f"  2. Move `{name}` from TEMPLATES to IMPLEMENTED in sonair/cli.py, naming the\n"
        f"     module that registers its losses/metrics.\n"
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
    backend = _selected_backend(argv)
    if backend is not None:
        argv += ["--scope", f"framework={backend}"]


def _selected_backend(argv: List[str]) -> Optional[str]:
    """The backend verb in ``argv``, or ``None`` (``sonair --help`` and friends)."""
    return next((arg for arg in argv[1:] if arg in BACKENDS), None)


def _pin_keras_backend(backend: Optional[str]) -> None:
    """``sonair tensorflow …`` must pin ``KERAS_BACKEND`` before ANYTHING imports keras.

    Keras 3 reads the variable once, at import time, and ``recordstream.keras`` defaults it to
    the first backend INSTALLED — torch, on any machine that also carries ``sonair[lightning]``.
    A torch-backed ``keras.Model`` cannot be trained with ``tf.GradientTape``, so the default has
    to be overridden for this verb.

    It happens HERE, in ``main()``, rather than inside the handler, because the handler runs too
    late: liquifai resolves the config first, and a ``!class:sonair.TensorFlowClassifier`` tag
    imports the runnable — and therefore keras — during that resolution.
    :mod:`sonair._tensorflow` sets the same default as a second line of defence for callers that
    never go through this CLI (``recordstream run cfg.yaml --scope framework=tensorflow``), and
    ``TensorFlowClassifier.initialize()`` refuses to run on the wrong engine either way.

    ``setdefault``, so an explicit ``KERAS_BACKEND=…`` still wins and is then reported as a
    mismatch instead of being silently overridden.
    """
    if backend == "tensorflow":
        os.environ.setdefault("KERAS_BACKEND", "tensorflow")


def main() -> None:
    _pin_keras_backend(_selected_backend(sys.argv))
    _inject_framework_scope(sys.argv)
    app.run()


if __name__ == "__main__":
    main()
