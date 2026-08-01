"""Sonair CLI — image classification across training backends.

Three verbs, and the training backend is a *config* dimension the CLI passes
through::

    sonair train    cfg.yaml                      # the config's default backend
    sonair train    cfg.yaml --framework keras    # a `!scope:framework=keras` block
    sonair evaluate cfg.yaml --framework torch
    sonair predict  cfg.yaml --framework tensorflow
    sonair frameworks                             # what the backends are

What stays constant across backends is the point of the demo: the same
``runnable:`` + ``task:`` config, the same three verbs, and the same logging
backends. Only the runnable class named in the YAML changes. (A generic
``recordstream run cfg.yaml --framework keras`` also works — it calls ``.run()``
with whatever ``task`` the YAML declares, with no verb at all.)

**Why the backend is a flag and not a subcommand.** It used to be one — the retired
form was ``sonair <backend> <verb> cfg.yaml``, eight backends x three verbs = 24
commands. But a backend verb never *selected* anything: it appended
``--scope framework=<backend>`` to ``sys.argv`` and the config's scope block did the
work. Two consequences made that worth undoing. Nothing compared the verb against
the config, so naming a backend the config had no block for trained the DEFAULT one
silently; and the sub-apps differed in exactly two values, which is a dict. liquifai
already binds ``--KEY VAL`` for any dimension a config declares, so the flag needed no
new machinery — and confluid now raises when a scope value matches no block, which is
the check the verb could never do. See ``docs/architecture.md`` §1b.

Adding a backend: write ``sonair/<name>_classifier.py`` (see ``docs/frameworks.md``),
then add a row to ``FRAMEWORKS`` below.
"""

import os
import sys
from typing import Any, Callable, Dict, List, NamedTuple, Optional

from liquifai import LiquifyApp
from loggair import get_logger

logger = get_logger(__name__)

app = LiquifyApp(name="sonair", description="Image classification across training backends")


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


# --------------------------------------------------------------------------- #
# The backends
# --------------------------------------------------------------------------- #
# Two values each, which is why this is a dict and not a registry module: what to
# say about the backend, and which module registers its losses/metrics under a
# plain NAME. Per-backend BEHAVIOUR stays out of it — `_pin_keras_backend` below
# would be a third column that is `None` seven times out of eight.


class Framework(NamedTuple):
    """One training backend: what to say about it, what to import for it."""

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


FRAMEWORKS: Dict[str, Framework] = {
    "lightning": Framework(
        "PyTorch Lightning — the reference backend (timm models, Lightning loggers).",
        "sonair.torch_classification",
    ),
    # Shares `lightning`'s loss/metric set, and that is the point rather than an economy: a
    # `torch.nn` loss does not care whether a Trainer or a hand-written loop calls it, so the
    # two backends differ ONLY in the loop.
    "torch": Framework(
        "PyTorch — a hand-written loop, no training framework at all (the control experiment).",
        "sonair.torch_classification",
    ),
    "keras": Framework(
        "Keras 3 — one API over PyTorch / TensorFlow / JAX (set KERAS_BACKEND to pick).",
        "sonair.keras_classification",
    ),
    "tensorflow": Framework(
        "TensorFlow — a native tf.data + tf.GradientTape loop, with SavedModel export.",
        "sonair.keras_classification",
    ),
    "fastai": Framework(
        "fastai — a Learner with one-cycle scheduling, over the same torch stack.",
        "sonair.fastai_classification",
    ),
    "huggingface": Framework(
        "Transformers `Trainer` + Accelerate — fine-tuning a Hub checkpoint.",
        # No registrations: the model computes its own loss (see the Framework field note).
    ),
    "jax": Framework(
        "JAX — a jitted optax loop over a Flax model (functional transforms).",
        # No registrations: optax losses are plain FUNCTIONS, so a config names them.
    ),
    "mlx": Framework(
        "MLX — training on Apple-silicon unified memory (macOS arm64 only).",
        # No registrations: `mlx.nn.losses` are plain FUNCTIONS, so a config names them.
    ),
}

#: Verb -> the ``task`` it forces. The verb wins over any ``task:`` in the YAML.
VERB_TASKS = {"train": "fit", "evaluate": "test", "predict": "predict"}

_VERB_DOCS = {
    "train": "Train an image classifier",
    "evaluate": "Evaluate a trained classifier over the test set",
    "predict": "Stream predictions from a trained classifier to its sink",
}


def _verb_command(verb: str) -> Callable[[Any], None]:
    """One verb.

    A factory, not a default-argument closure: liquifai reads the handler's SIGNATURE to
    build the command, so smuggling the loop variable in as a default would advertise it
    as a ``--_verb`` option in help and TAB completion. The signature it must expose is
    exactly ``(runnable)`` — the parameter NAME is what liquifai's DI binds to the
    config's ``runnable:`` key.
    """

    def command(runnable: Any) -> None:
        _dispatch(runnable, VERB_TASKS[verb])

    command.__name__ = verb
    command.__doc__ = f"{_VERB_DOCS[verb]} (task={VERB_TASKS[verb]}). Pick a backend with --framework."
    return command


for _verb in VERB_TASKS:
    app.script_command(name=_verb, flow_mode="manual")(_verb_command(_verb))


@app.command(name="frameworks")
def frameworks() -> None:
    """List the training backends `--framework` accepts."""
    width = max(len(name) for name in FRAMEWORKS)
    print("Training backends — pass one as `--framework <name>`:\n")
    for name, spec in FRAMEWORKS.items():
        print(f"  {name:<{width}}  {spec.description}")
    print(
        "\nThe name must also match a `!scope:framework=<name>` block in the config;"
        "\nconfluid lists the values a config declares when it does not."
    )


def _register_classification(framework: Optional[str]) -> None:
    """Import a backend's loss/metric registrations so a config may NAME them.

    A ``!class:`` target that is a registered name (``CrossEntropyLoss``, or a selector
    like ``SparseCategoricalCrossentropy@role=loss``) resolves only once the module that
    registers it has been imported; a dotted path resolves through importlib either way.
    Discovery services bootstrap every ``confluid.configurables`` entry point, but a CLI
    does not.

    It happens from ``main()``, before ``app.run()``, because the EAGER instance form
    (``!class:Foo()`` with parens) is built during config load — earlier than any handler
    runs. With no ``--framework`` this imports nothing: the flag is what names the engine,
    and there is no safe guess (importing the default's set would drag torch into a
    keras-only install). A config that names no engine-specific set, or spells its losses
    as dotted paths, needs none of this.

    Best-effort: a missing backend is a normal state on a partial install, and its absent
    registrations are the correct outcome, not an error.
    """
    if framework is None:
        return
    module = FRAMEWORKS.get(framework, Framework("")).registrations
    if module is None:
        return
    import importlib

    try:
        importlib.import_module(module)
    except ImportError as exc:
        logger.debug(f"{framework}: loss/metric registrations unavailable ({module}: {exc})")


def _normalize_framework_flag(argv: List[str]) -> None:
    """Rewrite ``--framework <name>`` into the ``--scope framework=<name>`` liquifai reads.

    liquifai binds ``--KEY VAL`` to scope activation only for dimensions the config
    DECLARES; against a config with no ``!scope:framework=…`` block at all, a bare
    ``--framework keras`` would instead be read as a config override and try to set a
    ``framework`` attribute on something. Rewriting here makes the flag mean one thing
    regardless of what the config happens to declare — an undeclared dimension stays an
    inert no-op, which is what lets a config grow into a backend it does not have yet.

    Mutates ``argv`` in place because scopes are parsed from the command line in phase 3,
    so this must land before ``run()``. The explicit ``--scope framework=…`` spelling is
    left exactly as typed.
    """
    out: List[str] = []
    i, n = 0, len(argv)
    while i < n:
        arg = argv[i]
        if arg.startswith("--framework="):
            out += ["--scope", f"framework={arg.split('=', 1)[1]}"]
        elif arg == "--framework" and i + 1 < n:
            out += ["--scope", f"framework={argv[i + 1]}"]
            i += 1
        else:
            out.append(arg)
        i += 1
    argv[:] = out


def _requested_framework(argv: List[str]) -> Optional[str]:
    """The backend named on the command line, in any spelling, or ``None``.

    Read from ``argv`` rather than from liquifai because both callers need it BEFORE the
    config is loaded — see :func:`_pin_keras_backend` and :func:`_register_classification`.
    """
    for i, arg in enumerate(argv):
        if arg.startswith("--framework="):
            return arg.split("=", 1)[1]
        if arg.startswith(("--scope=", "-s=")) and "framework=" in arg:
            return arg.split("framework=", 1)[1]
        if arg in ("--framework", "--scope", "-s") and i + 1 < len(argv):
            following = argv[i + 1]
            if arg == "--framework":
                return following
            if following.startswith("framework="):
                return following.split("=", 1)[1]
    return None


def _pin_keras_backend(framework: Optional[str]) -> None:
    """``--framework tensorflow`` must pin ``KERAS_BACKEND`` before ANYTHING imports keras.

    Keras 3 reads the variable once, at import time, and ``recordstream.keras`` defaults it to
    the first backend INSTALLED — torch, on any machine that also carries ``sonair[lightning]``.
    A torch-backed ``keras.Model`` cannot be trained with ``tf.GradientTape``, so the default has
    to be overridden for this backend.

    It happens HERE, in ``main()``, rather than inside the handler, because the handler runs too
    late: liquifai resolves the config first, and a ``!class:sonair.TensorFlowClassifier`` tag
    imports the runnable — and therefore keras — during that resolution.
    :mod:`sonair._tensorflow` sets the same default as a second line of defence for callers that
    never go through this CLI (``recordstream run cfg.yaml --framework tensorflow``), and
    ``TensorFlowClassifier.initialize()`` refuses to run on the wrong engine either way.

    ``setdefault``, so an explicit ``KERAS_BACKEND=…`` still wins and is then reported as a
    mismatch instead of being silently overridden.
    """
    if framework == "tensorflow":
        os.environ.setdefault("KERAS_BACKEND", "tensorflow")


def main() -> None:
    framework = _requested_framework(sys.argv)
    _pin_keras_backend(framework)
    _register_classification(framework)
    _normalize_framework_flag(sys.argv)
    app.run()


if __name__ == "__main__":
    main()
