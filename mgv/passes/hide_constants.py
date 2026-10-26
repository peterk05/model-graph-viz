"""Constant hiding pass (brief section 3.3.2).

Before:  Input --in 0--> MatMul <--in 1-- Constant (fp32 [1024, 768])
After:   Input --in 0--> MatMul          MatMul's label shows "[1]: fp32[1024, 768]"

The text goes into the consumer's visualization metadata under ``input_annotations``,
keyed by the consumer's input port, so the original port index is preserved.
"""

from __future__ import annotations

from mgv.graph import ROLE_CONSTANT, VIZ_INPUT_ANNOTATIONS, Graph, format_tensor
from mgv.passes.base import TransformationPass, register_pass, split_list


@register_pass
class HideConstantsPass(TransformationPass):
    """Hide constant nodes and show their data type and shape on the operations that use them."""

    name = "hide-constants"
    description = ("Hide Constant nodes; show their dtype/shape as attributes of the consuming "
                   "operations ('[port]: dtype[shape]', plus '(shared: <id>)' for shared constants).")
    option_help = {
        "keep-for": "comma-separated op types whose constant inputs stay visible (default: Result)",
    }

    def __init__(self, keep_for: list[str] | None = None) -> None:
        # A constant that feeds a model output IS the output, so by default that
        # connection stays drawn. Such a constant keeps a visible edge and is not removed.
        self.keep_for = set(keep_for) if keep_for is not None else {"Result"}

    def apply(self, graph: Graph) -> Graph:
        # Collect the ids first: we remove nodes while looping, and changing a dict
        # while looping over it is an error in Python.
        constant_ids = [node.id for node in graph.nodes.values() if node.role == ROLE_CONSTANT]

        for constant_id in constant_ids:
            out_edges = graph.outputs_of(constant_id)
            # "Shared" means the constant feeds more than one consumer input. A constant
            # used twice by the same op (e.g. Add(c, c)) counts as shared too.
            shared = len(out_edges) > 1

            for edge in out_edges:
                consumer = graph.nodes[edge.dst]
                if consumer.type in self.keep_for:
                    continue
                text = format_tensor(edge.dtype, edge.shape)
                if shared:
                    text += f" (shared: {constant_id})"
                consumer.viz.setdefault(VIZ_INPUT_ANNOTATIONS, {})[edge.dst_port] = text
                graph.remove_edge(edge.id)

            # Remove the constant only when it has no visible connections left.
            if not graph.out_edges[constant_id] and not graph.in_edges[constant_id]:
                graph.remove_node(constant_id)
        return graph

    @classmethod
    def from_options(cls, options: dict[str, str]) -> HideConstantsPass:
        cls.check_option_names(options)
        if "keep-for" in options:
            return cls(keep_for=split_list(options["keep-for"]))
        return cls()

    def __repr__(self) -> str:
        return f"HideConstantsPass(keep_for={sorted(self.keep_for)})"
