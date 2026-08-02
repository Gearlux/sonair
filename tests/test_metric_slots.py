"""The per-split metric slots: which backend serves which, and that all of them share ONE rule.

``metrics:`` is what every split measures; ``<split>_metrics:`` is what that split measures on
top. The RULE itself lives in ``marainer.metric_slots`` and is pinned by marainer's own suite —
what belongs here is the fleet: which slots each engine can actually serve, and that no backend
re-derives the combination inline.

The fleet-wide checks read the SOURCE with ``ast`` rather than importing, for the reason
``test_tracking_parity.py`` gives: no venv carries every backend's framework, so an import-based
check silently reduces to whichever extras happen to be installed — exactly the machine where a
copy-pasted omission lands.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Set

import pytest

SOURCE_DIR = Path(__file__).resolve().parent.parent / "sonair"

#: Which per-split slots each backend's engine can actually SERVE. The asymmetry is the point:
#: a slot is offered where the engine can compute it and absent where it cannot, rather than
#: accepted everywhere and silently ignored by half the fleet.
#:
#: * all three — the loop belongs to this project, so each split gets its own metric objects.
#: * val + test only — the engine's metric hook IS its evaluation hook (a fastai ``Learner``
#:   validates, a ``transformers.Trainer`` calls ``compute_metrics`` on eval, and the JAX / MLX
#:   training passes compute the loss and nothing else), so a train-only metric has nowhere to go.
#: * keras takes all three and computes their UNION — ``compile()`` owns one list.
SERVED = {
    "lightning_classifier": {"train", "val", "test"},
    "torch_classifier": {"train", "val", "test"},
    "tensorflow_classifier": {"train", "val", "test"},
    "keras_classifier": {"train", "val", "test"},
    "fastai_classifier": {"val", "test"},
    "jax_classifier": {"val", "test"},
    "mlx_classifier": {"val", "test"},
    "huggingface_classifier": {"val", "test"},
}


def constructor_params(stem: str) -> Set[str]:
    """The declared ``__init__`` parameter names of the module's classifier class."""
    tree = ast.parse((SOURCE_DIR / f"{stem}.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            args = node.args
            return {a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)}
    raise AssertionError(f"{stem}: no __init__ found")


# ---------------------------------------------------------------------------
# The fleet declares the slots its engine can serve
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("stem", sorted(SERVED))
def test_every_backend_declares_the_split_slots_it_can_serve(stem: str) -> None:
    params = constructor_params(stem)
    for split in SERVED[stem]:
        assert f"{split}_metrics" in params, f"{stem} must declare {split}_metrics as a ctor param"


@pytest.mark.parametrize("stem", sorted(SERVED))
def test_no_backend_offers_a_slot_its_engine_cannot_compute(stem: str) -> None:
    """The asymmetry is deliberate, so it is pinned in both directions.

    A `train_metrics` param on a backend whose training pass computes no metrics would accept a
    wired metric set and silently never call it — the exact failure mode the split slots exist
    to remove.
    """
    params = constructor_params(stem)
    for split in {"train", "val", "test"} - SERVED[stem]:
        assert f"{split}_metrics" not in params, (
            f"{stem} declares {split}_metrics, but its engine has no {split} metric hook. "
            f"Either serve it or do not offer it."
        )


@pytest.mark.parametrize("stem", sorted(SERVED))
def test_every_backend_uses_the_shared_combination_rule(stem: str) -> None:
    """One rule, applied to each engine's own vocabulary — never re-derived per backend.

    A backend that re-implemented the merge inline would drift from the others, and a config
    moving between them would silently change which metrics it computes.
    """
    source = (SOURCE_DIR / f"{stem}.py").read_text()
    called = {
        (node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", ""))
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
    }
    assert called & {"merge_metric_specs", "union_metric_specs"}, f"{stem} does not use the shared rule"


def test_the_checked_list_is_every_backend_on_disk() -> None:
    """The check that stops a ninth backend from quietly skipping all of the above."""
    on_disk = {p.stem for p in SOURCE_DIR.glob("*_classifier.py")}
    assert on_disk == set(SERVED), f"backends on disk but unchecked: {on_disk - set(SERVED)}"
