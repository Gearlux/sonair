"""Sonair CLI — image-classification training and evaluation.

``sonair train`` / ``sonair evaluate`` / ``sonair predict`` bind the single
:class:`~sonair.classifier.LightningClassifier` (a standard Lightning module) and
run it with the matching ``task``. The verb wins over any ``task:`` set in YAML, so
one config can serve all three. (A generic ``recordstream run config.yaml`` also
works — it just calls ``.run()`` with whatever ``task`` the YAML declares.)
"""

from typing import Any

from liquifai import LiquifyApp
from loggair import get_logger

logger = get_logger(__name__)

app = LiquifyApp(name="sonair", description="Sonair image-classification commands")


def _dispatch(classifier: Any, task: str) -> None:
    if classifier is None:
        logger.error("No classifier was bound — provide one under 'runnable:' in your YAML.")
        return
    from confluid import flow
    from confluid.fluid import Fluid

    if isinstance(classifier, Fluid):
        classifier = flow(classifier)
    if hasattr(classifier, "task"):
        classifier.task = task  # the verb wins over the YAML's task
    classifier.run()


@app.script_command(flow_mode="auto")
def train(runnable: Any) -> None:
    """Train an image classifier (task=fit)."""
    _dispatch(runnable, "fit")


@app.script_command(flow_mode="auto")
def evaluate(runnable: Any) -> None:
    """Evaluate a trained image classifier over the test set (task=test)."""
    _dispatch(runnable, "test")


@app.script_command(flow_mode="auto")
def predict(runnable: Any) -> None:
    """Stream predictions from a trained classifier to its sink (task=predict)."""
    _dispatch(runnable, "predict")


def main() -> None:
    app.run()


if __name__ == "__main__":
    main()
