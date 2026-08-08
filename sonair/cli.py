"""Sonair CLI — image classification across training backends.

Three verbs, and the training backend is a *config* dimension the CLI passes
through::

    sonair train    cfg.yaml                      # the config's default backend
    sonair train    cfg.yaml --framework keras    # a `!scope:framework=keras` block
    sonair evaluate cfg.yaml --framework torch
    sonair predict  cfg.yaml --framework tensorflow
    sonair frameworks                             # what the backends are

What stays constant across backends is the point of the demo: the same
``runnable:`` + ``task:`` config, the same three verbs, and the same tracking.
Only the runnable class named in the YAML changes. (A generic
``matrainer train cfg.yaml --framework keras`` or ``recordstream run cfg.yaml``
also works — the first is the same machinery without this project's table.)

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

The MACHINERY is :mod:`matrainer.cli` (2026-08-07, the P4 lift) — the verb dispatch
through ``materialize_runnable``, the ``--framework`` → ``--scope`` rewrite, the keras
engine pin, the registration import and their load-bearing ORDER all live on the
factory's own docstrings. What this module owns is sonair's DATA: the FRAMEWORKS
table and the verb help texts.

Adding a backend: write ``sonair/<name>_classifier.py`` (see ``docs/frameworks.md``),
then add a row to ``FRAMEWORKS`` below.
"""

from typing import Dict

from matrainer.cli import Framework, make_trainer_app, trainer_main

# --------------------------------------------------------------------------- #
# The backends
# --------------------------------------------------------------------------- #
# Two values each, which is why this is a dict and not a registry module: what to
# say about the backend, and which module registers its losses/metrics under a
# plain NAME. Per-backend BEHAVIOUR stays out of it — the keras engine pin lives in
# the shared machinery, where a third column here would be `None` seven times out
# of eight.
#
# `tensorflow` shares the Keras registration set on purpose — in TF >= 2.16
# `tf.keras` IS Keras 3, so a separate `tensorflow_classification.py` would
# register the same classes twice and make every bare
# `!class:SparseCategoricalCrossentropy` ambiguous for no gain.
#
# `None` means the backend has NOTHING to register, which is a real state rather
# than an oversight: a `transformers` model computes its own loss from
# `config.problem_type`, so there is no loss a config could wire
# (`docs/architecture.md` §5).

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
        # No registrations: the model computes its own loss (see the table note above).
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

app = make_trainer_app(
    name="sonair",
    description="Image classification across training backends",
    frameworks=FRAMEWORKS,
    verb_docs={
        "train": "Train an image classifier",
        "evaluate": "Evaluate a trained classifier over the test set",
        "predict": "Stream predictions from a trained classifier to its sink",
    },
)


def main() -> None:
    trainer_main(app, FRAMEWORKS)


if __name__ == "__main__":
    main()
