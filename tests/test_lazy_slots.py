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

import inspect
import re
from typing import Set

from confluid import lazy_param_names

from sonair.lightning_classifier import LightningClassifier


def _slots_flowed_at_runtime() -> Set[str]:
    """``self.<name>`` passed to ``flow(...)`` anywhere in the runnable's module."""
    module = inspect.getmodule(LightningClassifier)
    assert module is not None
    source = inspect.getsource(module)
    return set(re.findall(r"flow\(self\.([a-z_]+)", source))


def test_every_flowed_slot_is_declared_lazy() -> None:
    flowed = _slots_flowed_at_runtime()
    assert flowed, "found no flow(self.<slot>) calls — has the dispatch changed?"

    undeclared = sorted(flowed - lazy_param_names(LightningClassifier))

    assert not undeclared, (
        f"{undeclared} are flowed at run time but not annotated Lazy[T]. "
        "An auto-flow walker would build them eagerly — for a slot needing a runtime "
        "kwarg, that raises; for the rest it silently discards the deferral."
    )


def test_the_lazy_scan_sees_this_class_at_all() -> None:
    """Guard: the body scan reads ``__init__`` SOURCE, so a packaged deployment sees none.

    Without this, the assertion above would pass vacuously wherever the scan is blind.
    """
    assert lazy_param_names(LightningClassifier), "no lazy slots detected — the body scan may be blind here"
