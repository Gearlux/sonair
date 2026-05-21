"""Sonair marainer plugin — mounts ``marainer classification`` sub-app.

This module is discovered by marainer's plugin system via the
``marainer.commands`` entry-point group (declared in ``sonair/pyproject.toml``).
It adds a ``classification`` command group to the running marainer app,
exposing:

* ``marainer classification train``    — alias to :func:`sonair.cli.train`
* ``marainer classification evaluate`` — alias to :func:`sonair.cli.evaluate`
"""

from liquifai import LiquifyApp

from sonair.cli import evaluate, train


def register(app: LiquifyApp) -> None:
    """Mount the ``classification`` sub-app onto *app* (the root marainer app)."""
    classification = LiquifyApp(
        name="classification",
        description="Image-classification workflow commands (sonair specialisation).",
    )
    classification.script_command(flow_mode="auto")(train)
    classification.script_command(flow_mode="auto")(evaluate)
    app.add_app(classification, name="classification")
