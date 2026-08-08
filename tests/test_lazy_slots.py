"""Every slot the runnable flows at run time must DECLARE itself deferred.

``Lazy[T]`` is what tells a deep-flow walker to leave a slot alone until the runnable
flows it with the kwarg that does not exist yet (``params=``, ``dataset=``,
``num_classes=``). Annotating such a slot ``Any`` — or hand-writing
``Union[X, Fluid]``, which is the ``Lazy`` alias minus its marker — declares nothing,
and nothing catches it: the class keeps working only because the slot happens to hold a
``LazyClass`` value.

This derives the expectation from the SOURCE rather than restating a list, so a slot
added tomorrow is covered without editing this file.
"""

import ast
import inspect
import re
from pathlib import Path
from typing import Dict, Set

from confluid import lazy_param_names

from sonair.lightning_classifier import LightningClassifier

_PACKAGE = Path(__file__).resolve().parent.parent / "sonair"


def _mro_source() -> str:
    """The runnable's WHOLE implementation: its module plus every base's, workspace-owned only.

    Since the 2026-08-06 consolidation the machinery lives in shared bases (matrainer's
    trainer API + this project's task bases), so the flowed-slot ↔ declared-lazy pairing must
    be checked over the union — a module-local scan would find nothing and pass vacuously.
    """
    chunks, seen = [], set()
    for klass in LightningClassifier.__mro__:
        module = inspect.getmodule(klass)
        if module is None or module.__name__ in seen:
            continue
        if module.__name__.split(".")[0] not in {"sonair", "matrainer"}:
            continue
        seen.add(module.__name__)
        chunks.append(inspect.getsource(module))
    return "\n".join(chunks)


def test_every_flowed_slot_is_declared_lazy() -> None:
    source = _mro_source()
    flowed = set(re.findall(r"flow\(\s*self\.([a-z_]+)", source))
    assert flowed, "found no flow(self.<slot>) calls anywhere in the MRO — has the dispatch changed?"

    undeclared = sorted(flowed - (_declared_lazy_slots(source) | lazy_param_names(LightningClassifier)))

    assert not undeclared, (
        f"{undeclared} are flowed at run time but not annotated Lazy[T]. "
        "An auto-flow walker would build them eagerly — for a slot needing a runtime "
        "kwarg, that raises; for the rest it silently discards the deferral."
    )


def _declared_lazy_slots(source: str) -> Set[str]:
    """Slot names a module annotates ``Lazy[...]`` — as a ctor param OR an ``__init__`` body attr."""
    params = set(re.findall(r"\b([a-z_]+): (?:Mandatory\[)?(?:Optional\[)?Lazy\[", source))
    body = set(re.findall(r"self\.([a-z_]+): (?:Optional\[)?Lazy\[", source))
    return params | body


def test_every_backend_declares_the_slots_it_flows() -> None:
    """The same rule as above, applied to ALL EIGHT backends by reading their SOURCE.

    ``lazy_param_names`` needs the class IMPORTED, and seven of the eight need a framework a
    default install lacks — so an import-based sweep would skip exactly where a copy-pasted slot
    lands. A regex over the module text covers every backend on every machine.
    """
    offenders: Dict[str, list] = {}
    checked = 0
    for path in sorted(_PACKAGE.glob("*_classifier.py")):
        source = path.read_text()
        flowed = set(re.findall(r"flow\(\s*self\.([a-z_]+)", source))
        if not flowed:
            # A module with no flow calls must be one built on the shared bases (the machinery
            # — and its flows — live there; covered by the MRO-union test above).
            assert any(
                m in source
                for m in (
                    "ClassificationTorchHooks",
                    "ClassificationKerasHooks",
                    "HuggingFaceRunnable",
                    "JaxLoopRunnable",
                    "MlxLoopRunnable",
                )
            ), f"{path.name}: no flow(self.<slot>) calls and no shared base — has the dispatch changed?"
            checked += 1
            continue

        undeclared = sorted(flowed - _declared_lazy_slots(source))
        if undeclared:
            offenders[path.name] = undeclared
        checked += 1

    assert checked >= 8, f"expected every backend to be scanned, saw {checked}"
    assert not offenders, (
        f"{offenders} are flowed at run time but not annotated Lazy[T]. "
        "An auto-flow walker would build them eagerly — for a slot needing a runtime "
        "kwarg (or a positional runtime arg) that raises; for the rest it silently "
        "discards the deferral."
    )


def _model_slot_annotations() -> Dict[str, str]:
    """``{module filename: source text of the ``model`` ctor param's annotation}``.

    Read from SOURCE with ``ast`` rather than by importing: six of the seven backends need a
    framework this workspace does not install by default, so an import-based check would skip
    exactly where a copy-pasted slot would land.
    """
    found: Dict[str, str] = {}
    for path in sorted(_PACKAGE.glob("*_classifier.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.FunctionDef) and node.name == "__init__":
                for arg in node.args.args:
                    if arg.arg == "model" and arg.annotation is not None:
                        found[path.name] = ast.unparse(arg.annotation)
    return found


def test_no_classifier_declares_its_model_slot_as_lazy_any() -> None:
    """``Lazy[Any]`` declares the DEFERRAL and nothing else — half a declaration.

    The marker is what a deep-flow walker reads, so ``Lazy[Any]`` "works"; what it costs is the
    other half. A reader cannot tell what the slot admits, mypy checks nothing through it, and
    ``to_pydantic`` hands navigaitor / StreamStudio an open field where a real type (or a union
    of the two the backend accepts) would have described the contract. Each backend names what
    it actually takes — its framework's own module type (``torch.nn.Module``, ``keras.Model``,
    ``nnx.Module``, ``mlx.nn.Module``) or a spelled-out union where two genuinely meet.
    """
    annotations = _model_slot_annotations()
    assert len(annotations) >= 7, f"expected every backend to declare a `model` slot, found {annotations}"

    undeclared = sorted(name for name, anno in annotations.items() if "Lazy[" not in anno)
    untyped = sorted(name for name, anno in annotations.items() if "Lazy[Any]" in anno.replace(" ", ""))

    assert not undeclared, f"{undeclared} do not declare `model` deferred — an auto-flow walker would build it"
    assert not untyped, (
        f"{untyped} declare `model` as `Lazy[Any]`. Name the type the backend actually takes — "
        "a union when two frameworks genuinely meet in the slot (see `TensorFlowNetwork`)."
    )


def test_the_lazy_scan_sees_this_class_at_all() -> None:
    """Guard: the body scan reads ``__init__`` SOURCE, so a packaged deployment sees none.

    Without this, the assertion above would pass vacuously wherever the scan is blind.
    """
    assert lazy_param_names(LightningClassifier), "no lazy slots detected — the body scan may be blind here"
