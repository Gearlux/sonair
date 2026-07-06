"""Sonair CLI — image-classification training and evaluation commands.

Provides ``sonair train`` and ``sonair evaluate`` as aliases to the
classification-specialised Lightning trainer/evaluator. These are equivalent
to ``marainer classification train`` / ``marainer classification evaluate``
(which are registered via the marainer plugin in :mod:`sonair.plugin`).
"""

from liquifai import LiquifyApp
from loggair import get_logger

from sonair.evaluator import LightningClassificationEvaluator
from sonair.trainer import LightningClassificationTrainer

logger = get_logger(__name__)

app = LiquifyApp(name="sonair", description="Sonair image-classification commands")


@app.script_command(flow_mode="auto")
def train(trainer: LightningClassificationTrainer) -> None:
    """Train an image classifier."""
    trainer.run()


@app.script_command(flow_mode="auto")
def evaluate(evaluator: LightningClassificationEvaluator) -> None:
    """Evaluate a trained image classifier."""
    evaluator.run()


def main() -> None:
    app.run()


if __name__ == "__main__":
    main()
