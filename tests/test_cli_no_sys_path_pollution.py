"""Regression: sonair.cli MUST NOT inject cwd ('.') into sys.path.

The workspace mandate (taidal/CLAUDE.md, "No sys.path Hacks") forbids
``sys.path.insert(0, ".")`` in CLI entry points. In a monorepo where
project folders match installed package names, this causes
``multiprocessing.spawn`` DataLoader workers (``num_workers > 0``) to
crash with::

    ImportError: cannot import name 'LiquifyApp' from 'liquifai'
    (unknown location)

The chain: parent imports ``sonair.cli`` via matrainer's plugin loader →
the module-level ``sys.path.insert(0, ".")`` pollutes the parent's
sys.path → ``multiprocessing.spawn.get_preparation_data`` copies that
polluted sys.path to each worker → the worker's PathFinder finds
``<workspace>/liquifai/`` (project dir, no ``__init__.py``) as a PEP 420
namespace portion BEFORE the editable finder runs → ``liquifai`` resolves
to a namespace package with no exports.

Fixed in commit 2c55707 (sonair). This test guards against re-introduction.
"""

import subprocess
import sys


def test_sonair_cli_import_does_not_add_dot_to_sys_path() -> None:
    """Importing ``sonair.cli`` in a fresh interpreter MUST leave ``'.'`` out of sys.path."""
    code = (
        "import sys\n"
        "import sonair.cli  # noqa: F401 — side-effect import is the point\n"
        "assert '.' not in sys.path, (\n"
        "    f\"sonair.cli polluted sys.path with '.' — re-introduces \"\n"
        "    f'multiprocessing.spawn DataLoader worker shadowing. '\n"
        "    f'See CLAUDE.md No sys.path Hacks mandate. sys.path={sys.path!r}'\n"
        ")\n"
        "print('CLEAN')\n"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        "sonair.cli import polluted sys.path with '.'. See CLAUDE.md "
        "'No sys.path Hacks' mandate and commit 2c55707.\n"
        f"stderr:\n{result.stderr}"
    )
    assert result.stdout.strip().endswith("CLEAN")
