"""The transformation-pass interface and the registry of available passes.

A pass takes a Graph and changes it (adds visualization metadata, removes nodes,
rewires edges) and returns it. Passes only ever see the framework-independent Graph,
never OpenVINO objects, so they cannot modify the original model.

Who copies the graph? The Pipeline copies the parsed graph ONCE before running any
pass, then every pass edits that copy in place. That keeps the caller's graph
unchanged without making a full copy of a 5,000-node graph for every single pass.
If you call a pass on its own and want to keep the input, call ``graph.copy()`` first.

To add a new pass: subclass ``TransformationPass``, give it a ``name``, implement
``apply`` and (if it has options) ``from_options``, and decorate it with
``@register_pass``. The parser, renderer and CLI do not need to change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from mgv.graph import Graph


class PassConfigError(Exception):
    """A pass name or pass option is invalid."""


class TransformationPass(ABC):
    #: Name used to select the pass on the command line, e.g. "hide-constants".
    name: ClassVar[str] = ""
    #: One-line description shown by ``--list-passes``.
    description: ClassVar[str] = ""
    #: Option names this pass accepts, mapped to a short help text.
    option_help: ClassVar[dict[str, str]] = {}

    @abstractmethod
    def apply(self, graph: Graph) -> Graph:
        """Transform ``graph`` in place and return it."""

    @classmethod
    def from_options(cls, options: dict[str, str]) -> TransformationPass:
        """Build the pass from string options (as given on the command line).

        Passes without options can use this default. Passes with options override it.
        """
        if options:
            raise PassConfigError(f"pass {cls.name!r} takes no options, got: {', '.join(sorted(options))}")
        return cls()

    @classmethod
    def check_option_names(cls, options: dict[str, str]) -> None:
        unknown = sorted(set(options) - set(cls.option_help))
        if unknown:
            allowed = ", ".join(sorted(cls.option_help)) or "none"
            raise PassConfigError(
                f"unknown option(s) for pass {cls.name!r}: {', '.join(unknown)} (allowed: {allowed})"
            )

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


# pass name -> pass class
_PASSES: dict[str, type[TransformationPass]] = {}


def register_pass(pass_cls: type[TransformationPass]) -> type[TransformationPass]:
    """Class decorator: make a pass selectable by its name."""
    if not pass_cls.name:
        raise PassConfigError(f"{pass_cls.__name__} must set a 'name'")
    if pass_cls.name in _PASSES and _PASSES[pass_cls.name] is not pass_cls:
        raise PassConfigError(f"a pass named {pass_cls.name!r} is already registered")
    _PASSES[pass_cls.name] = pass_cls
    return pass_cls


def available_passes() -> dict[str, type[TransformationPass]]:
    return dict(sorted(_PASSES.items()))


def create_pass(name: str, options: dict[str, str] | None = None) -> TransformationPass:
    """Create a registered pass by name, configured from string options."""
    pass_cls = _PASSES.get(name)
    if pass_cls is None:
        known = ", ".join(sorted(_PASSES)) or "none"
        raise PassConfigError(f"unknown pass {name!r} (available: {known})")
    return pass_cls.from_options(options or {})


def split_list(text: str) -> list[str]:
    """Turn 'a, b,c' into ['a', 'b', 'c'], ignoring empty items."""
    return [item.strip() for item in text.split(",") if item.strip()]
