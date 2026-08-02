"""The sweep overlay resolves to the right runnable + model for every cell of its matrix.

`examples/sweep/all_backends.yaml` is an overlay: it `include:`s the example and overrides only
what a sweep needs. Two things about that composition are easy to get wrong and silent when
wrong, and both are checked here rather than discovered thirteen training runs later.

**A scope block REPLACES, it does not merge.** `include:` deep-merges two *mappings*, and a
`!scope:` block is a `ScopeBlock` object rather than a dict — so an overlay block with a key the
included file already uses discards that block wholesale. A `keras:` block holding only the
nested convnet dropped the runnable, the model and the preprocessing chain, and
`--framework keras` resolved to `LightningClassifier` + `TimmModel`: the wrong thing trained
under the right name, with nothing raising.

**A nested scope must compose with its parent.** `--framework mlx --model convnet` has to reach
`MlxConvNet`, not the example's top-level `TorchConvNet` — which is what the whole per-backend
convnet arrangement exists for.

Config-level on purpose, like `test_cli_frameworks.py`: no venv has all eight frameworks
installed, so flowing the classes would reduce the check to whichever extras happen to be
present — exactly the machine where a copy-pasted block lands.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import confluid
import pytest

SWEEP = Path(__file__).resolve().parent.parent / "examples" / "sweep" / "all_backends.yaml"

#: (backend, variant) -> (runnable class, model class). The matrix the driver script walks.
#: `None` for a variant an engine structurally cannot have — jax and mlx wrap no zoo, and a
#: `transformers.Trainer` drives models that compute their own loss, so it takes no convnet.
MATRIX: Dict[Tuple[str, str], Tuple[str, str]] = {
    ("lightning", "zoo"): ("LightningClassifier", "TimmModel"),
    ("lightning", "convnet"): ("LightningClassifier", "TorchConvNet"),
    ("torch", "zoo"): ("TorchClassifier", "TimmModel"),
    ("torch", "convnet"): ("TorchClassifier", "TorchConvNet"),
    ("fastai", "zoo"): ("FastaiClassifier", "TimmModel"),
    ("fastai", "convnet"): ("FastaiClassifier", "TorchConvNet"),
    ("keras", "zoo"): ("KerasClassifier", "KerasModel"),
    ("keras", "convnet"): ("KerasClassifier", "KerasConvNet"),
    ("tensorflow", "zoo"): ("TensorFlowClassifier", "KerasModel"),
    ("tensorflow", "convnet"): ("TensorFlowClassifier", "TensorFlowConvNet"),
    ("huggingface", "zoo"): ("HuggingFaceClassifier", "HuggingFaceModel"),
    ("jax", "convnet"): ("JaxClassifier", "FlaxConvNet"),
    ("mlx", "convnet"): ("MlxClassifier", "MlxConvNet"),
}

TRACKERS = {
    "mlflow": "MlflowTracker",
    "tensorboard": "TensorBoardTracker",
    "csv": "CsvTracker",
    "aim": "AimTracker",
    "clearml": "ClearMLTracker",
    "wandb": "WandbTracker",
}


def resolve(framework: str, variant: str = "zoo", tracker: str = "mlflow") -> Dict[str, Any]:
    scopes = [f"framework={framework}", f"tracker={tracker}"]
    if variant == "convnet":
        scopes.append("model=convnet")
    document = confluid.load(confluid.load_config(SWEEP), flow=False, scopes=scopes)
    assert isinstance(document, dict)
    return document


def target_name(value: Any) -> Optional[str]:
    """The class name behind a `!class:` / `!lazy:` marker, without importing it."""
    target = getattr(value, "target", None)
    if target is None:
        return None
    return str(target).rsplit(".", 1)[-1]


@pytest.mark.parametrize("cell", sorted(MATRIX), ids=lambda c: f"{c[0]}-{c[1]}")
def test_every_backend_resolves_to_its_own_runnable_and_model(cell: Tuple[str, str]) -> None:
    """The regression pin for the replace-not-merge trap described in the module docstring."""
    framework, variant = cell
    expected_runnable, expected_model = MATRIX[cell]
    document = resolve(framework, variant)

    assert target_name(document["runnable"]) == expected_runnable
    assert target_name(document.get("model")) == expected_model


def test_the_torch_backed_backends_share_one_convnet() -> None:
    """Three backends, one `TorchConvNet` — the example's top-level block, inherited unchanged.

    If one of them ever resolves to something else, a per-backend block has been added where
    the shared one was doing the job, which is the duplication `convnet/spec.py` exists to avoid.
    """
    names = {target_name(resolve(fw, "convnet")["model"]) for fw in ("lightning", "torch", "fastai")}
    assert names == {"TorchConvNet"}


def test_each_non_torch_engine_gets_its_own_implementation_of_the_shared_spec() -> None:
    """The point of the sweep: the SAME architecture, in each framework's own module type."""
    assert {fw: target_name(resolve(fw, "convnet")["model"]) for fw in ("keras", "tensorflow", "jax", "mlx")} == {
        "keras": "KerasConvNet",
        "tensorflow": "TensorFlowConvNet",
        "jax": "FlaxConvNet",
        "mlx": "MlxConvNet",
    }


@pytest.mark.parametrize("tracker", sorted(TRACKERS))
def test_each_tracker_scope_selects_exactly_that_backend(tracker: str) -> None:
    trackers = resolve("lightning", tracker=tracker)["trackers"]

    assert [target_name(t) for t in trackers] == [TRACKERS[tracker]]


@pytest.mark.parametrize("tracker", sorted(TRACKERS))
def test_the_tracker_takes_its_names_from_the_run(tracker: str) -> None:
    """`!ref:` rather than a literal, so the ONE pair the CLI sets reaches the tracker too.

    A literal here would drift from whatever the run is actually called — the one thing a sweep
    must not do, since the tracker is where the runs are compared.
    """
    document = resolve("lightning", tracker=tracker)
    wired = document["trackers"][0]

    for key in ("experiment_name", "run_name"):
        assert type(wired.kwargs[key]).__name__ == "Reference", f"{tracker}: {key} is not a !ref:"


def test_a_metric_set_never_reaches_an_engine_that_cannot_consume_it() -> None:
    """Same rule as the example: torchmetrics are `framework="torch"`-bound.

    A block can escape the top-level set two ways, and both are in use here, so this checks the
    EFFECTIVE value rather than the document key. Either the block nulls the key (`keras`,
    `tensorflow` — their runnable takes the metrics through broadcasting, so the key has to go),
    or it names its own on the runnable (`fastai`, `jax`, `mlx`, `huggingface` — an explicit
    constructor kwarg wins over a broadcast one, verified, so the inherited key has no reader).

    `val_metrics` has no second escape: every one of those runnables grew that slot with the
    per-split metrics work, so an un-nulled key WOULD reach it.
    """
    for framework in ("keras", "tensorflow"):
        document = resolve(framework)
        assert not document.get("metrics"), f"{framework} inherited the torchmetrics set"

    for framework in ("fastai", "jax", "mlx", "huggingface"):
        wired = resolve(framework)["runnable"].kwargs.get("metrics")
        assert wired, f"{framework} must name its own metrics — the inherited ones are torchmetrics"
        assert all(isinstance(m, str) for m in wired), f"{framework}: engine metrics are NAMES, not classes"

    for framework in ("keras", "tensorflow", "fastai", "jax", "mlx", "huggingface"):
        assert not resolve(framework).get("val_metrics"), f"{framework} inherited the torchmetrics val set"

    for framework in ("lightning", "torch"):
        document = resolve(framework)
        assert document["metrics"], f"{framework} runs on torch and should keep the shared metrics"
        assert "confusion" in document["val_metrics"]
