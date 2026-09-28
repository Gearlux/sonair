"""The distribution metadata a PyPI upload actually requires.

Every rule here guards a failure the build, the test suite, the linters and the
examples all pass straight through — measured on this project's own wheel before
its first release, when ``twine check --strict`` refused it:

* a dependency written as a direct URL makes PyPI *reject the upload* — the file
  is uploaded, parsed, and refused, so the failure arrives after everything green;
* a ``readme``/``license-files`` entry naming a file that is not there fails at
  build time on a clean checkout but not in an editable install;
* missing readme/classifiers/URLs cost nothing at build time and produce a landing
  page with no description, no license, no home, and no issue tracker.

The rules read ``pyproject.toml``; nothing is built. Whether every extra RESOLVES
against PyPI needs the network, so the release workflow checks that instead
(``.github/workflows/release.yml``). The sibling ``test_docs_links.py`` covers the
README's *contents* once it becomes the landing page.
"""

import re
import tomllib
from pathlib import Path
from typing import Any, Dict, List, cast

_REPO = Path(__file__).resolve().parent.parent
_PYPROJECT = _REPO / "pyproject.toml"


def _project() -> Dict[str, Any]:
    with _PYPROJECT.open("rb") as fh:
        return cast(Dict[str, Any], tomllib.load(fh)["project"])


def _every_requirement() -> List[str]:
    """Core dependencies plus every extra's — PyPI validates all of them alike."""
    project = _project()
    requirements = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        requirements.extend(extra)
    return requirements


def test_no_requirement_is_a_direct_url() -> None:
    """A ``name @ git+https://...`` requirement is refused by PyPI at upload time.

    Warehouse validates every ``Requires-Dist`` and rejects any whose requirement
    carries a URL. It applies to extras exactly as to core dependencies, which is the
    trap: an extra nobody installs still blocks the whole upload.
    """
    direct = [req for req in _every_requirement() if "@" in req and "://" in req.split("@", 1)[1]]

    assert not direct, f"PyPI rejects direct-URL requirements: {direct}"


def test_the_declared_readme_and_license_files_exist() -> None:
    """``readme`` / ``license-files`` name files that are actually in the tree.

    Both are read at BUILD time, so an editable install never notices a missing one;
    the failure surfaces on the clean checkout that builds the release artifact.
    """
    project = _project()
    declared = [project["readme"], *project.get("license-files", [])]
    missing = [name for name in declared if not (_REPO / name).exists()]

    assert not missing, f"pyproject names files that do not exist: {missing}"


def test_the_landing_page_metadata_is_present() -> None:
    """Description, readme, license, classifiers and URLs — the PyPI page's whole frame.

    Each is optional to the build, which is why all of them were missing at once:
    the wheel built identically without them and ``twine check --strict`` was the
    first thing to notice (``long_description missing``).
    """
    project = _project()
    missing = [key for key in ("description", "readme", "license", "classifiers", "urls") if not project.get(key)]

    assert not missing, f"release metadata missing from pyproject: {missing}"


def test_the_development_status_classifier_matches_the_version() -> None:
    """An ``aN``/``bN`` version says pre-release; the classifier must say so too.

    These drift in opposite directions — the version moves every release, the
    classifier is written once and forgotten — leaving a "Production/Stable" badge
    on an alpha.
    """
    project = _project()
    status = [c for c in project["classifiers"] if c.startswith("Development Status ::")]
    assert len(status) == 1, f"expected exactly one Development Status classifier, got {status}"

    is_prerelease = "a" in project["version"] or "b" in project["version"] or "rc" in project["version"]
    expected = "3 - Alpha" if is_prerelease else "4 - Beta"

    assert expected in status[0], f"version {project['version']} does not match classifier {status[0]!r}"


def test_the_changelog_is_shipped_in_the_sdist() -> None:
    """`MANIFEST.in` must carry `CHANGELOG.md` — nothing else pulls it in.

    setuptools builds an sdist from what is *referenced*: `readme` brings README.md,
    `license-files` brings LICENSE, `packages.find` + `package-data` bring the package.
    A CHANGELOG is referenced by nothing, so without `MANIFEST.in` it is silently absent.

    Asserted on the declaration rather than by building an sdist: building one takes
    seconds and needs a clean tree, and the declaration is what actually goes stale.
    """
    manifest = _REPO / "MANIFEST.in"
    assert manifest.exists(), "MANIFEST.in is missing; the sdist would ship no CHANGELOG"

    included = {
        line.split(maxsplit=1)[1].strip()
        for line in manifest.read_text().splitlines()
        if line.strip().startswith("include ")
    }
    missing = [name for name in ("CHANGELOG.md",) if name not in included]

    assert not missing, f"MANIFEST.in does not include: {missing}"


def test_the_newest_changelog_entry_is_the_declared_version() -> None:
    """The version being released has its CHANGELOG entry, and it is the top one.

    The version in `pyproject.toml` is bumped in one edit and the CHANGELOG in
    another; the release workflow checks the tag against the first and nothing
    checks the second, so a release whose notes describe the previous version
    would publish without complaint.
    """
    versions = re.findall(r"^## \[([^\]]+)\]", (_REPO / "CHANGELOG.md").read_text(), flags=re.MULTILINE)

    assert versions, "CHANGELOG.md has no `## [<version>]` entry"
    assert (
        versions[0] == _project()["version"]
    ), f"newest CHANGELOG entry is {versions[0]!r}, pyproject.toml declares {_project()['version']!r}"
