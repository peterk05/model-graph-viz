"""Graph -> Graphviz DOT -> SVG (brief section 3.4).

Each node is drawn as an HTML-like table:

    ┌────────────────────────────────┐
    │  in 0  │  in 1                 │   <- one cell per input port (edges attach here)
    │            MatMul              │   <- operation type (bold)
    │        name of the node        │
    │      transpose_b = true        │   <- operation attributes
    │     [1]: fp32[1024, 768]       │   <- hidden constant inputs (from hide-constants)
    │    out 0: fp32[?, ?, 2048]     │   <- one cell per output port, with dtype + shape
    └────────────────────────────────┘

Edges go from an output-port cell to an input-port cell, so the port numbers are
visible and every arrow attaches to the right socket. The ``dot`` engine lays the
graph out top-to-bottom in layers.

Why dtype/shape sit on the output port instead of on the edges: Graphviz turns every
edge label into an extra node in its layout, which made the simplified TinyLlama
graph take over 10 minutes. On the port cell the information is still visible (and
written once per tensor even when the tensor feeds several consumers), and each edge
also gets it as a hover tooltip. Edge labels can still be switched on for small graphs.
"""

from __future__ import annotations

import html
from pathlib import Path

import graphviz

from mgv.graph import (
    ROLE_CONSTANT,
    ROLE_INPUT,
    ROLE_OUTPUT,
    ROLE_REFERENCE,
    VIZ_BORDER_COLOR,
    VIZ_INPUT_ANNOTATIONS,
    VIZ_NOTES,
    Graph,
    Node,
    format_tensor,
)
from mgv.render.base import Renderer, RenderError

# Fill colours by role. Any op type not listed gets the default, so op types the
# renderer has never heard of still draw correctly.
_FILL_BY_ROLE = {
    ROLE_INPUT: "#d8f0d8",
    ROLE_OUTPUT: "#d8e4f8",
    ROLE_CONSTANT: "#ececec",
    ROLE_REFERENCE: "#fff6d5",
}
_DEFAULT_FILL = "#ffffff"
_PORT_FILL = "#f3f3f3"


class GraphvizRenderer(Renderer):
    def __init__(
        self,
        rankdir: str = "TB",
        show_edge_labels: bool = False,
        max_attributes: int = 6,
        max_text_length: int = 48,
        large_graph_threshold: int = 500,
        save_dot: bool = False,
    ) -> None:
        self.rankdir = rankdir                  # "TB" = top-to-bottom layered layout
        self.show_edge_labels = show_edge_labels
        self.max_attributes = max_attributes    # long attribute lists are cut, with "+N more"
        self.max_text_length = max_text_length  # long names/values are cut with "…"
        # Above this many nodes, cap Graphviz's slowest step (see to_dot).
        self.large_graph_threshold = large_graph_threshold
        self.save_dot = save_dot                # also write the .dot source next to the SVG

    # ---------- public ----------

    def to_dot(self, graph: Graph) -> graphviz.Digraph:
        dot = graphviz.Digraph(name=_safe_id(graph.name))
        dot.attr(rankdir=self.rankdir, newrank="true", nodesep="0.3", ranksep="0.45")
        if len(graph.nodes) > self.large_graph_threshold:
            # dot spends almost all its time on the horizontal positioning step
            # (network simplex). nslimit caps its iterations: a slightly less tidy
            # layout in exchange for minutes instead of tens of minutes.
            dot.attr(nslimit="1")
        dot.attr("node", shape="plain", fontname="Helvetica", fontsize="10")
        dot.attr("edge", color="#555555", arrowsize="0.6", fontname="Helvetica", fontsize="8")

        output_tensors = _output_tensor_text(graph)
        for node in graph.nodes.values():
            dot.node(_node_key(node.id), label=self._node_label(graph, node, output_tensors))

        for edge in graph.edges.values():
            tensor = format_tensor(edge.dtype, edge.shape)
            attributes = {"tooltip": f"out {edge.src_port} -> in {edge.dst_port}: {tensor}"}
            if self.show_edge_labels:
                attributes["label"] = tensor
            dot.edge(
                f"{_node_key(edge.src)}:out{edge.src_port}:s",
                f"{_node_key(edge.dst)}:in{edge.dst_port}:n",
                **attributes,
            )
        return dot

    def render(self, graph: Graph, output_path: Path) -> Path:
        output_path = Path(output_path)
        if output_path.suffix.lower() != ".svg":
            raise RenderError(f"output file must end in .svg, got {str(output_path)!r}")
        dot = self.to_dot(graph)
        try:
            svg_bytes = dot.pipe(format="svg")
        except graphviz.ExecutableNotFound as error:
            raise RenderError("Graphviz is not installed or 'dot' is not on PATH "
                              "(Ubuntu: sudo apt install graphviz)") from error
        except graphviz.CalledProcessError as error:
            raise RenderError(f"Graphviz failed to render the graph: {error}") from error
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(svg_bytes)
            if self.save_dot:
                output_path.with_suffix(".dot").write_text(dot.source, encoding="utf-8")
        except OSError as error:
            raise RenderError(f"cannot write {str(output_path)!r}: {error}") from error
        return output_path

    # ---------- labels ----------

    def _node_label(self, graph: Graph, node: Node, output_tensors: dict[tuple[int, int], str]) -> str:
        if node.role == ROLE_REFERENCE:
            return self._reference_label(node)
        n_in, n_out = graph.port_counts(node.id)
        border_color = node.viz.get(VIZ_BORDER_COLOR)
        border = "3" if border_color else "1"
        color = border_color or "#333333"
        fill = _FILL_BY_ROLE.get(node.role, _DEFAULT_FILL)

        rows: list[str] = []
        if n_in:
            rows.append(self._port_row("in", [f"in {i}" for i in range(n_in)]))
        rows.append(f"<TR><TD><B>{self._text(node.type)}</B></TD></TR>")
        if node.name and node.name != node.type:
            rows.append(f'<TR><TD><FONT POINT-SIZE="8" COLOR="#555555">{self._text(node.name)}</FONT></TD></TR>')

        attr_items = list(node.attrs.items())
        for key, value in attr_items[: self.max_attributes]:
            rows.append(f'<TR><TD><FONT POINT-SIZE="8">{self._text(f"{key} = {value}")}</FONT></TD></TR>')
        if len(attr_items) > self.max_attributes:
            more = len(attr_items) - self.max_attributes
            rows.append(f'<TR><TD><FONT POINT-SIZE="8" COLOR="#777777">+{more} more</FONT></TD></TR>')

        annotations: dict[int, str] = node.viz.get(VIZ_INPUT_ANNOTATIONS, {})
        for port in sorted(annotations):
            text = f"[{port}]: {annotations[port]}"
            rows.append(f'<TR><TD><FONT POINT-SIZE="8" COLOR="#1f4e79">{self._text(text, limit=80)}</FONT></TD></TR>')

        for note in node.viz.get(VIZ_NOTES, []):
            rows.append(f'<TR><TD><FONT POINT-SIZE="8" COLOR="#8a6d00">{self._text(note, limit=80)}</FONT></TD></TR>')

        if n_out:
            texts = []
            for port in range(n_out):
                tensor = output_tensors.get((node.id, port))
                texts.append(f"out {port}: {tensor}" if tensor else f"out {port}")
            rows.append(self._port_row("out", texts))

        return (f'<<TABLE BORDER="{border}" COLOR="{color}" CELLBORDER="0" CELLSPACING="0" '
                f'CELLPADDING="3" BGCOLOR="{fill}">' + "".join(rows) + "</TABLE>>")

    def _reference_label(self, node: Node) -> str:
        """A small dashed tag standing in for a far-away source node."""
        border_color = node.viz.get(VIZ_BORDER_COLOR, "#b08d00")
        return (f'<<TABLE BORDER="1" STYLE="dashed" COLOR="{border_color}" CELLBORDER="0" '
                f'CELLSPACING="0" CELLPADDING="2" BGCOLOR="{_FILL_BY_ROLE[ROLE_REFERENCE]}">'
                f'<TR><TD PORT="out0"><FONT POINT-SIZE="8">{self._text(node.name)}</FONT></TD></TR></TABLE>>')

    def _port_row(self, kind: str, texts: list[str]) -> str:
        """A row of small port cells; PORT="in0", PORT="out1" etc. are what edges attach to."""
        cells = "".join(
            f'<TD PORT="{kind}{index}" BORDER="1" COLOR="#aaaaaa" BGCOLOR="{_PORT_FILL}">'
            f'<FONT POINT-SIZE="7">{self._text(text)}</FONT></TD>'
            for index, text in enumerate(texts)
        )
        return ('<TR><TD><TABLE BORDER="0" CELLBORDER="0" CELLSPACING="2" CELLPADDING="1"><TR>'
                + cells + "</TR></TABLE></TD></TR>")

    def _text(self, text: str, limit: int | None = None) -> str:
        limit = limit or self.max_text_length
        if len(text) > limit:
            text = text[: limit - 1] + "…"
        return html.escape(text)


def _output_tensor_text(graph: Graph) -> dict[tuple[int, int], str]:
    """(node id, output port) -> "dtype[shape]", taken from the first edge leaving that port."""
    texts: dict[tuple[int, int], str] = {}
    for edge in graph.edges.values():
        key = (edge.src, edge.src_port)
        if key not in texts:
            texts[key] = format_tensor(edge.dtype, edge.shape)
    return texts


def _node_key(node_id: int) -> str:
    return f"n{node_id}"


def _safe_id(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name) or "graph"
