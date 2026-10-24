"""The parser interface and a registry that picks a parser by file extension.

To support a new model format, write a subclass of ``ModelParser`` and decorate it
with ``@register_parser``. Nothing else in the library has to change, because passes
and renderers only ever see the framework-independent ``Graph``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from mgv.graph import Graph


class ModelLoadError(Exception):
    """The model file could not be found, is not supported, or could not be read."""


class ModelParser(ABC):
    """Turns a model file into a framework-independent Graph."""

    #: File extensions this parser handles, lower case, with the dot (e.g. ``(".xml",)``).
    extensions: tuple[str, ...] = ()
    #: Short human-readable format name, used in error messages.
    format_name: str = "unknown"

    @abstractmethod
    def parse(self, path: Path) -> Graph:
        """Read the model at ``path`` and return a new Graph. Raise ModelLoadError on failure."""


# extension -> parser class
_PARSERS: dict[str, type[ModelParser]] = {}


def register_parser(parser_cls: type[ModelParser]) -> type[ModelParser]:
    """Class decorator: make a parser available for its file extensions."""
    for extension in parser_cls.extensions:
        _PARSERS[extension.lower()] = parser_cls
    return parser_cls


def supported_extensions() -> list[str]:
    return sorted(_PARSERS)


def get_parser(path: Path) -> ModelParser:
    """Return a parser instance for this file, chosen by its extension."""
    extension = path.suffix.lower()
    parser_cls = _PARSERS.get(extension)
    if parser_cls is None:
        known = ", ".join(supported_extensions()) or "none"
        raise ModelLoadError(
            f"unsupported model file {str(path)!r}: extension {extension or '(none)'!r} "
            f"is not supported (supported: {known})"
        )
    return parser_cls()


def load_model(path: str | Path) -> Graph:
    """Load any supported model file into a Graph."""
    model_path = Path(path)
    if not model_path.exists():
        raise ModelLoadError(f"model file not found: {str(model_path)!r}")
    if not model_path.is_file():
        raise ModelLoadError(f"model path is not a file: {str(model_path)!r}")
    return get_parser(model_path).parse(model_path)
