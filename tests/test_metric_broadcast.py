"""The torchmetrics registrations close their ``**kwargs`` accept-lists (``broadcast="declared"``).

Every torchmetrics class has the same constructor shape — explicit params plus
``**kwargs`` flowing up to :class:`torchmetrics.Metric`, which consumes a fixed
set (``compute_on_cpu`` …) and refuses the rest. Unregistered (or registered
open), that ``**kwargs`` makes confluid's accept-list ``None`` = accept
EVERYTHING, so every bare trainer key in a config (``batch_size:`` /
``max_epochs:`` / ``run_name:``) cascaded into every metric marker. Registered
``broadcast="declared"``, the accept-list is the parsed slot set — signature
params plus the ``Metric`` kwargs the body scan reads off its
``self.x = kwargs.pop("x", …)`` assignments — so metric knobs still sweep while
trainer keys stop landing.

The registration must fire in EVERY torch-family run, including a bare
``recordstream run`` that materializes a classifier without going through the
sonair CLI — which imports :mod:`sonair.classification_torch` (the shared hooks)
but not the registration module. The hooks module therefore imports
:mod:`sonair.torch_classification` itself; the first test pins that edge.
"""

import sys

from confluid import load, marks
from torchmetrics.classification import MulticlassF1Score

import sonair.classification_torch  # noqa: F401 — the import under test: it must pull the registrations in


def test_the_torch_hooks_module_pulls_in_the_metric_registrations() -> None:
    """Importing the shared hooks module alone must register the metric set,
    because that is the ONLY sonair module a bare ``recordstream run`` imports."""
    assert "sonair.torch_classification" in sys.modules
    assert marks(MulticlassF1Score).name == "MulticlassF1Score"
    assert marks(MulticlassF1Score).broadcast_declared is True


def test_bare_trainer_keys_do_not_reach_a_metric_marker() -> None:
    """The defect this closes: a training config's top-level keys swept into every
    metric's ``**kwargs`` surface."""
    settled = load(
        """
batch_size: 32
max_epochs: 3
run_name: efficientvit_b0_mnist
metrics:
  f1_macro: !partial:torchmetrics.classification.MulticlassF1Score
    average: macro
""",
        until="settled",
    )
    assert dict(settled["metrics"]["f1_macro"].kwargs) == {"average": "macro"}


def test_a_bare_metric_kwarg_still_sweeps_into_the_marker() -> None:
    """The reason for ``"declared"`` over ``broadcast=False``: a genuine ``Metric``
    kwarg written as a bare sweep key still reaches every metric."""
    settled = load(
        """
compute_on_cpu: true
batch_size: 32
metrics:
  f1_macro: !partial:torchmetrics.classification.MulticlassF1Score
    average: macro
""",
        until="settled",
    )
    assert dict(settled["metrics"]["f1_macro"].kwargs) == {"compute_on_cpu": True, "average": "macro"}
