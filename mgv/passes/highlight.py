"""Node highlighting pass (brief section 3.3.1)."""

from __future__ import annotations

from pathlib import Path

from mgv.graph import VIZ_BORDER_COLOR, Graph
from mgv.passes.base import PassConfigError, TransformationPass, register_pass, split_list


@register_pass
class HighlightNodesPass(TransformationPass):
    """Give every node whose name is in ``names`` a coloured border (red by default)."""

    name = "highlight"
    description = "Highlight nodes by name by changing their border colour (default red)."
    option_help = {
        "names": "comma-separated node names to highlight",
        "names-file": "text file with one node name per line",
        "color": "border colour, any Graphviz colour name or #rrggbb (default: red)",
    }

    def __init__(self, names: list[str], color: str = "red") -> None:
        self.names = set(names)
        self.color = color
        self.matched: set[str] = set()   # filled in by apply(), useful for warnings

    def apply(self, graph: Graph) -> Graph:
        self.matched = set()
        for node in graph.nodes.values():
            if node.name in self.names:
                node.viz[VIZ_BORDER_COLOR] = self.color
                self.matched.add(node.name)
        return graph

    @property
    def unmatched(self) -> set[str]:
        """Names that were asked for but not found in the last graph."""
        return self.names - self.matched

    @classmethod
    def from_options(cls, options: dict[str, str]) -> "HighlightNodesPass":
        cls.check_option_names(options)
        names = split_list(options.get("names", ""))
        if "names-file" in options:
            path = Path(options["names-file"])
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError as error:
                raise PassConfigError(f"cannot read names-file {str(path)!r}: {error}") from error
            names += [line.strip() for line in lines if line.strip()]
        if not names:
            raise PassConfigError("pass 'highlight' needs at least one node name: "
                                  "use highlight.names=NAME1,NAME2 or highlight.names-file=FILE")
        return cls(names, options.get("color", "red"))

    def __repr__(self) -> str:
        return f"HighlightNodesPass({len(self.names)} names, color={self.color!r})"
