"""Custom pass (brief section 3.3.3): detach high-fan-out tensors.

The problem
-----------
In TinyLlama a handful of tensors are computed once and used by all 22 decoder
layers: the RoPE sin/cos tables, the attention mask, ``beam_idx`` and the KV-cache
gather. In the drawing that becomes ~240 edges running from the top of the graph to
every layer, a curtain of parallel lines that hides the real computation flow (and
makes Graphviz's layout much slower, because every long edge is routed through every
layer in between).

The fix
-------
For every node output that feeds at least ``min-uses`` consumers, replace each of
those edges with a small *reference* node placed next to the consumer, e.g.
``↳ Unsqueeze #812 out 0``. The reference keeps the original consumer input port and
the tensor's dtype/shape, and the source node gets a note saying where its uses went.
The id after ``#`` is the source's unique node id, so the connection can always be
traced back.
"""

from __future__ import annotations

from mgv.graph import ROLE_REFERENCE, VIZ_NOTES, Graph
from mgv.passes.base import PassConfigError, TransformationPass, register_pass


@register_pass
class DetachFanoutPass(TransformationPass):
    name = "detach-fanout"
    description = ("Replace edges from tensors used by many consumers (e.g. masks, RoPE tables) "
                   "with small reference nodes next to each consumer.")
    option_help = {
        "min-uses": "detach outputs used by at least this many consumer inputs (default: 8)",
    }

    def __init__(self, min_uses: int = 8) -> None:
        if min_uses < 2:
            raise PassConfigError("detach-fanout: min-uses must be at least 2")
        self.min_uses = min_uses

    def apply(self, graph: Graph) -> Graph:
        # Count uses per (node id, output port) with one pass over the edges.
        uses: dict[tuple[int, int], list[int]] = {}
        for edge in graph.edges.values():
            uses.setdefault((edge.src, edge.src_port), []).append(edge.id)

        for (source_id, port), edge_ids in uses.items():
            if len(edge_ids) < self.min_uses:
                continue
            source = graph.nodes[source_id]
            for edge_id in edge_ids:
                edge = graph.edges[edge_id]
                reference = graph.add_node(
                    name=f"↳ {source.type} #{source_id} out {port}",
                    op_type="Reference",
                    attrs={},
                    role=ROLE_REFERENCE,
                    num_inputs=0,
                    num_outputs=1,
                )
                reference.viz["reference_to"] = source_id
                graph.add_edge(reference.id, 0, edge.dst, edge.dst_port, edge.shape, edge.dtype)
                graph.remove_edge(edge_id)
            source.viz.setdefault(VIZ_NOTES, []).append(
                f"out {port}: {len(edge_ids)} uses shown as ↳ #{source_id} references"
            )
        return graph

    @classmethod
    def from_options(cls, options: dict[str, str]) -> "DetachFanoutPass":
        cls.check_option_names(options)
        raw = options.get("min-uses", "8")
        try:
            return cls(min_uses=int(raw))
        except ValueError as error:
            raise PassConfigError(f"detach-fanout: min-uses must be a whole number, got {raw!r}") from error

    def __repr__(self) -> str:
        return f"DetachFanoutPass(min_uses={self.min_uses})"
