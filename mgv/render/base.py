"""The renderer interface. A renderer turns a Graph into an output file."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from mgv.graph import Graph


class RenderError(Exception):
    """The graph could not be rendered or the output could not be written."""


class Renderer(ABC):
    @abstractmethod
    def render(self, graph: Graph, output_path: Path) -> Path:
        """Draw ``graph`` into ``output_path`` and return the path written."""
