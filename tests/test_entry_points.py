"""Every module that registers a configurable is named in the entry-point group.

This exists because five backends forgot. Importing `sonair` registers NOTHING — every backend
name in `sonair/__init__.py` is lazy so a Keras-only install never drags in torch — so discovery
runs off `[project.entry-points."confluid.configurables"]` instead: a scanner imports each module
named there, and the import is what executes the `@configurable` decorators.

A module missing from that list therefore registers nothing, and the failure is SILENT in every
way a developer would notice. A dotted path in YAML (`!class:sonair.tensorflow_model.
TensorFlowModel`) imports the module itself, so configs keep working; the CLI imports its own
backend's module, so runs keep working. Only the discovery surfaces go blind — navigaitor's slot
pickers, the form-specs, the StreamStudio palette — and nothing fails, the class is just never
offered. That is exactly what happened to the huggingface / jax / mlx / tensorflow-model work of
2026-07-31: eight registering modules, no entry points, and a green suite either way.

So the expectation is DERIVED from the source tree rather than restated as a list here — a list
would need the same maintenance the entry points do, and would go stale in the same commit.

The scan is static (`ast`), deliberately: most backends are not installed in any one venv, so
importing to find out what registers would reduce this to whatever the local extras happen to be.
"""

import ast
import tomllib
from pathlib import Path
from typing import Dict, Optional, Set

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "sonair"
GROUP = "confluid.configurables"

#: Calls that register a class the registry can later resolve. `configurable(...)` is the
#: decorator applied as a function — how the loss/metric modules tag UPSTREAM classes they do
#: not own (`torch.nn.CrossEntropyLoss` cannot carry a decorator) — and `register` is confluid's
#: direct form. A module doing either needs an entry point just as much as a decorated one.
REGISTERING_CALLS = frozenset({"configurable", "register", "register_class"})


def _called_name(node: ast.expr) -> Optional[str]:
    """The bare name of a callable, whether it is `configurable` or `confluid.configurable`."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _registers_configurables(path: Path) -> bool:
    """Does this module put anything into confluid's registry when imported?

    Two spellings count, and both are in use here: a class carrying `@configurable(...)`, and a
    call to `configurable(...)` / `register(...)` for a class from another package. Note this
    reads the SOURCE — a bare mention inside a docstring or an error message does not count,
    which is why `sonair/cli.py` (whose template text quotes the decorator) is correctly absent.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for decorator in node.decorator_list:
                target = decorator.func if isinstance(decorator, ast.Call) else decorator
                if _called_name(target) == "configurable":
                    return True
        if isinstance(node, ast.Call) and _called_name(node.func) in REGISTERING_CALLS:
            return True
    return False


def _module_path(path: Path) -> str:
    """`sonair/convnet/keras_convnet.py` -> `sonair.convnet.keras_convnet`."""
    return ".".join(path.relative_to(PROJECT_ROOT).with_suffix("").parts)


def _registering_modules() -> Set[str]:
    """Every module in the package that registers something, found by reading the tree."""
    return {
        _module_path(path)
        for path in sorted(PACKAGE_ROOT.rglob("*.py"))
        if path.name != "__init__.py" and _registers_configurables(path)
    }


def _declared_entry_points() -> Dict[str, str]:
    """The group as `pyproject.toml` declares it — name -> module path.

    Read from the FILE, not from installed metadata. Metadata only refreshes on an editable
    reinstall, so checking it would fail for everyone who edited `pyproject.toml` and has not
    reinstalled yet — an environment condition, not the drift this pins.
    """
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as handle:
        data = tomllib.load(handle)
    entry_points: Dict[str, str] = data["project"]["entry-points"][GROUP]
    return entry_points


def test_every_registering_module_has_an_entry_point() -> None:
    """The one that would have caught the 2026-07-31 gap."""
    missing = _registering_modules() - set(_declared_entry_points().values())

    assert not missing, (
        "these modules register configurables but no discovery service will ever import them — "
        f'add a line each to [project.entry-points."{GROUP}"] in pyproject.toml: {sorted(missing)}'
    )


def test_every_declared_entry_point_names_a_real_module() -> None:
    """The other direction: a typo'd path is a no-op that looks like coverage.

    A scanner catches the ImportError per module and warns, which is the RIGHT behaviour for a
    backend that is merely not installed — and it means a misspelled path is indistinguishable
    from an absent framework. Checked on disk so it holds without the framework installed.
    """
    for name, module in sorted(_declared_entry_points().items()):
        parts = module.split(".")
        as_module = PROJECT_ROOT.joinpath(*parts).with_suffix(".py")
        as_package = PROJECT_ROOT.joinpath(*parts, "__init__.py")

        assert (
            as_module.exists() or as_package.exists()
        ), f"entry point {name!r} names a module that does not exist: {module}"


def test_the_group_is_modules_not_objects() -> None:
    """`module:attr` would register one name and hide the module's other classes.

    confluid's scanner walks a MODULE. Pointing an entry point at an attribute is accepted by
    the packaging layer and quietly narrows discovery to whatever that one object is.
    """
    for name, value in sorted(_declared_entry_points().items()):
        assert ":" not in value, f"entry point {name!r} names an object, not a module: {value}"
