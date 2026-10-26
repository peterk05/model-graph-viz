"""Custom pass (brief section 3.3.3): fold weight-decompression Converts.

The problem
-----------
TinyLlama exported with ``--weight-format fp16`` stores every weight as an fp16
Constant followed by a ``Convert`` to fp32:

    Constant (fp16 [2048, 2048]) --> Convert (to f32) --> MatMul

That is 388 extra Convert boxes in the graph. ``hide-constants`` alone cannot remove
them: the Constant's consumer is the Convert, so the Convert box stays on screen,
now with its own annotation, still sitting in front of every MatMul.

The fix
-------
For every Convert whose only input comes from a constant, connect the constant
directly to each of the Convert's consumers (same consumer input ports) and remove
the Convert. The new edge's data type records both types, e.g. ``fp16→fp32``.

Run it BEFORE ``hide-constants``; the MatMul then shows
``[1]: fp16→fp32[2048, 2048]`` and the decompression boxes disappear.
"""

from __future__ import annotations

from mgv.graph import ROLE_CONSTANT, Graph
from mgv.passes.base import TransformationPass, register_pass, split_list


@register_pass
class FoldConstantConvertsPass(TransformationPass):
    name = "fold-converts"
    description = ("Remove Convert ops that only decompress a constant (e.g. fp16 weights to fp32) "
                   "and connect the constant straight to the consumers. Run before hide-constants.")
    option_help = {
        "types": "comma-separated op types to fold (default: Convert)",
    }

    def __init__(self, types: list[str] | None = None) -> None:
        self.types = set(types) if types is not None else {"Convert"}

    def apply(self, graph: Graph) -> Graph:
        candidate_ids = [node.id for node in graph.nodes.values() if node.type in self.types]

        for convert_id in candidate_ids:
            in_edges = graph.inputs_of(convert_id)
            if len(in_edges) != 1:
                continue
            source = in_edges[0]
            if graph.nodes[source.src].role != ROLE_CONSTANT:
                continue   # a Convert on real activations is real computation: keep it

            for out_edge in graph.outputs_of(convert_id):
                graph.add_edge(
                    src=source.src,
                    src_port=source.src_port,
                    dst=out_edge.dst,
                    dst_port=out_edge.dst_port,   # keep the consumer's original input port
                    shape=out_edge.shape,
                    dtype=_combined_dtype(source.dtype, out_edge.dtype),
                )
            graph.remove_node(convert_id)   # also removes the Convert's old edges
        return graph

    @classmethod
    def from_options(cls, options: dict[str, str]) -> FoldConstantConvertsPass:
        cls.check_option_names(options)
        if "types" in options:
            return cls(types=split_list(options["types"]))
        return cls()

    def __repr__(self) -> str:
        return f"FoldConstantConvertsPass(types={sorted(self.types)})"


def _combined_dtype(stored: str | None, converted: str | None) -> str | None:
    if stored is None or converted is None or stored == converted:
        return converted if converted is not None else stored
    return f"{stored}→{converted}"
