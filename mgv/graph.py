"""Framework-independent graph: nodes are operations, edges connect ports.

Nothing in this file knows about OpenVINO (or any other framework).
A parser builds a Graph, transformation passes change it, and a renderer draws it.

Design notes
------------
* Every node gets a unique integer id from the Graph. Names are kept separately
  because real models can contain several nodes with the same name.
* Every edge also gets a unique id. That way two edges between the same pair of
  nodes (for example ``Add(x, x)``) are two separate objects and never merge.
* ``in_edges`` / ``out_edges`` map a node id to the ids of its edges. They are
  index tables that make "who feeds this node?" an O(1) lookup instead of a scan
  over every edge in the graph.
"""

from __future__ import annotations

from typing import Any

# Values for Node.role. Parsers translate their own framework's op types into these,
# so passes can say "is this a constant?" without knowing OpenVINO type names.
ROLE_OP = "op"
ROLE_INPUT = "input"
ROLE_OUTPUT = "output"
ROLE_CONSTANT = "constant"
# A visualization-only stand-in for another node (see the detach-fanout pass).
ROLE_REFERENCE = "reference"
ROLES = (ROLE_OP, ROLE_INPUT, ROLE_OUTPUT, ROLE_CONSTANT, ROLE_REFERENCE)

# Keys used in Node.viz (visualization metadata). Passes write them, renderers read them.
VIZ_BORDER_COLOR = "border_color"            # str, e.g. "red"
VIZ_INPUT_ANNOTATIONS = "input_annotations"  # dict[int, str]: input port -> text, e.g. {1: "fp32[1024, 768]"}
VIZ_NOTES = "notes"                          # list[str]: extra lines to show on the node


class GraphError(Exception):
    """Raised when a graph operation would make the graph invalid."""


class Node:
    """One operation in the graph."""

    def __init__(
        self,
        node_id: int,
        name: str,
        op_type: str,
        attrs: dict[str, str] | None = None,
        role: str = ROLE_OP,
        num_inputs: int | None = None,
        num_outputs: int | None = None,
    ) -> None:
        if role not in ROLES:
            raise GraphError(f"unknown node role {role!r}; expected one of {ROLES}")
        self.id = node_id              # unique, given by the Graph (names can repeat)
        self.name = name
        self.type = op_type            # e.g. "MatMul", "Constant", "Parameter"
        self.role = role               # "op", "input", "output" or "constant"
        self.attrs = attrs if attrs is not None else {}   # operation attributes, e.g. {"transpose_b": "true"}
        self.num_inputs = num_inputs   # number of input ports, if the parser knows it
        self.num_outputs = num_outputs # number of output ports, if the parser knows it
        # Visualization metadata written by passes and read by the renderer, e.g.
        # {"border_color": "red"} or {"input_annotations": {1: "fp32[1024, 768]"}}.
        self.viz: dict[str, Any] = {}

    def __repr__(self) -> str:
        return f"Node({self.id}, {self.name!r}, {self.type})"


class Edge:
    """A connection from one output port to one input port."""

    def __init__(
        self,
        edge_id: int,
        src: int,
        src_port: int,
        dst: int,
        dst_port: int,
        shape: list[str] | None = None,
        dtype: str | None = None,
    ) -> None:
        self.id = edge_id              # unique, so two edges between the same nodes stay separate
        self.src = src                 # source node id
        self.src_port = src_port       # which output of the source
        self.dst = dst                 # destination node id
        self.dst_port = dst_port       # which input of the destination
        self.shape = shape             # e.g. ["1", "?", "2048"]; "?" = dynamic dim; None = unknown
        self.dtype = dtype             # e.g. "fp16"; None = unknown

    def __repr__(self) -> str:
        return f"Edge({self.src}:{self.src_port} -> {self.dst}:{self.dst_port})"


def format_tensor(dtype: str | None, shape: list[str] | None) -> str:
    """Format tensor metadata the way the brief shows it, e.g. ``fp32[1024, 768]``.

    Missing data type -> ``?``. Unknown shape (even the rank is unknown) -> ``[...]``.
    A scalar has an empty shape -> ``fp32[]``.
    """
    dtype_text = dtype if dtype is not None else "?"
    if shape is None:
        return f"{dtype_text}[...]"
    return f"{dtype_text}[{', '.join(shape)}]"


class Graph:
    """A directed graph of operations connected port-to-port."""

    def __init__(self, name: str = "graph") -> None:
        self.name = name
        self.nodes: dict[int, Node] = {}           # node id -> Node
        self.edges: dict[int, Edge] = {}           # edge id -> Edge
        self.in_edges: dict[int, list[int]] = {}   # node id -> ids of edges coming in
        self.out_edges: dict[int, list[int]] = {}  # node id -> ids of edges going out
        self.next_node_id = 0
        self.next_edge_id = 0

    # ---------- building ----------

    def add_node(
        self,
        name: str,
        op_type: str,
        attrs: dict[str, str] | None = None,
        role: str = ROLE_OP,
        num_inputs: int | None = None,
        num_outputs: int | None = None,
    ) -> Node:
        node = Node(self.next_node_id, name, op_type, attrs, role, num_inputs, num_outputs)
        self.next_node_id += 1
        self.nodes[node.id] = node
        self.in_edges[node.id] = []
        self.out_edges[node.id] = []
        return node

    def add_edge(
        self,
        src: int,
        src_port: int,
        dst: int,
        dst_port: int,
        shape: list[str] | None = None,
        dtype: str | None = None,
    ) -> Edge:
        if src not in self.nodes:
            raise GraphError(f"unknown source node id {src}")
        if dst not in self.nodes:
            raise GraphError(f"unknown destination node id {dst}")
        if src_port < 0 or dst_port < 0:
            raise GraphError(f"port indices must be >= 0, got {src_port} and {dst_port}")
        edge = Edge(self.next_edge_id, src, src_port, dst, dst_port, shape, dtype)
        self.next_edge_id += 1
        self.edges[edge.id] = edge
        self.out_edges[src].append(edge.id)
        self.in_edges[dst].append(edge.id)
        return edge

    # ---------- removing ----------

    def remove_edge(self, edge_id: int) -> None:
        edge = self.edges.pop(edge_id)
        self.out_edges[edge.src].remove(edge_id)
        self.in_edges[edge.dst].remove(edge_id)

    def remove_node(self, node_id: int) -> None:
        """Remove a node and every edge touching it, so no edge points at a missing node."""
        for edge_id in self.in_edges[node_id] + self.out_edges[node_id]:
            self.remove_edge(edge_id)
        del self.nodes[node_id]
        del self.in_edges[node_id]
        del self.out_edges[node_id]

    # ---------- looking things up ----------

    def get_node(self, node_id: int) -> Node:
        return self.nodes[node_id]

    def inputs_of(self, node_id: int) -> list[Edge]:
        """Edges coming into a node, sorted by input port."""
        edges = [self.edges[e] for e in self.in_edges[node_id]]
        edges.sort(key=lambda edge: edge.dst_port)
        return edges

    def outputs_of(self, node_id: int) -> list[Edge]:
        """Edges going out of a node, sorted by output port."""
        edges = [self.edges[e] for e in self.out_edges[node_id]]
        edges.sort(key=lambda edge: edge.src_port)
        return edges

    def nodes_by_name(self, name: str) -> list[Node]:
        """All nodes with this name (names are not unique, so this returns a list)."""
        return [node for node in self.nodes.values() if node.name == name]

    def of_type(self, op_type: str) -> list[Node]:
        return [node for node in self.nodes.values() if node.type == op_type]

    def of_role(self, role: str) -> list[Node]:
        return [node for node in self.nodes.values() if node.role == role]

    def count_types(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for node in self.nodes.values():
            counts[node.type] = counts.get(node.type, 0) + 1
        return counts

    def port_counts(self, node_id: int) -> tuple[int, int]:
        """(number of input ports, number of output ports) to draw for a node.

        Uses what the parser recorded; if that is missing (hand-built graphs) it falls
        back to the highest port index actually used by an edge.
        """
        node = self.nodes[node_id]
        used_in = [self.edges[e].dst_port + 1 for e in self.in_edges[node_id]]
        used_out = [self.edges[e].src_port + 1 for e in self.out_edges[node_id]]
        n_in = max([node.num_inputs or 0] + used_in)
        n_out = max([node.num_outputs or 0] + used_out)
        return n_in, n_out

    # ---------- checking ----------

    def validate(self) -> None:
        """Raise GraphError if the edge tables and index tables disagree."""
        for edge_id, edge in self.edges.items():
            if edge.src not in self.nodes or edge.dst not in self.nodes:
                raise GraphError(f"edge {edge_id} points at a missing node: {edge}")
            if edge_id not in self.out_edges[edge.src] or edge_id not in self.in_edges[edge.dst]:
                raise GraphError(f"edge {edge_id} is missing from the index tables")
        for node_id in self.nodes:
            for edge_id in self.in_edges[node_id]:
                if self.edges[edge_id].dst != node_id:
                    raise GraphError(f"in_edges[{node_id}] lists edge {edge_id} that ends elsewhere")
            for edge_id in self.out_edges[node_id]:
                if self.edges[edge_id].src != node_id:
                    raise GraphError(f"out_edges[{node_id}] lists edge {edge_id} that starts elsewhere")
        if set(self.in_edges) != set(self.nodes) or set(self.out_edges) != set(self.nodes):
            raise GraphError("index tables and node table have different node ids")

    # ---------- copying ----------

    def copy(self) -> "Graph":
        """Return an independent copy, so changes to it never touch this graph."""
        new = Graph(self.name)
        for node in self.nodes.values():
            new_node = Node(node.id, node.name, node.type, dict(node.attrs), node.role,
                            node.num_inputs, node.num_outputs)
            new_node.viz = _copy_viz(node.viz)
            new.nodes[node.id] = new_node
            new.in_edges[node.id] = list(self.in_edges[node.id])
            new.out_edges[node.id] = list(self.out_edges[node.id])
        for edge in self.edges.values():
            shape = list(edge.shape) if edge.shape is not None else None
            new.edges[edge.id] = Edge(edge.id, edge.src, edge.src_port, edge.dst, edge.dst_port,
                                      shape, edge.dtype)
        new.next_node_id = self.next_node_id
        new.next_edge_id = self.next_edge_id
        return new

    def __repr__(self) -> str:
        return f"Graph({len(self.nodes)} nodes, {len(self.edges)} edges)"


def _copy_viz(viz: dict[str, Any]) -> dict[str, Any]:
    """Copy visualization metadata one level deep (values are strings, dicts or lists)."""
    copied: dict[str, Any] = {}
    for key, value in viz.items():
        if isinstance(value, dict):
            copied[key] = dict(value)
        elif isinstance(value, list):
            copied[key] = list(value)
        else:
            copied[key] = value
    return copied
