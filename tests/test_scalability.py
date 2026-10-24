"""Scalability: passes on a synthetic graph with more than 5,000 nodes stay fast.

The graph imitates a transformer: 40 'layers', each with weight constants feeding
MatMuls, a chain of activations, and a mask tensor shared by every layer.
"""

import time

from mgv.graph import ROLE_CONSTANT, ROLE_INPUT, Graph
from mgv.passes import DetachFanoutPass, FoldConstantConvertsPass, HideConstantsPass, HighlightNodesPass
from mgv.pipeline import Pipeline
from mgv.render import GraphvizRenderer


def build_large_graph(layers: int = 40, ops_per_layer: int = 45) -> Graph:
    graph = Graph("large")
    previous = graph.add_node("input", "Parameter", role=ROLE_INPUT).id
    mask = graph.add_node("mask", "Select").id
    graph.add_edge(previous, 0, mask, 0, ["1", "?"], "bool")
    for layer in range(layers):
        for i in range(ops_per_layer):
            weight = graph.add_node(f"l{layer}.w{i}", "Constant", role=ROLE_CONSTANT).id
            convert = graph.add_node(f"l{layer}.w{i}/convert", "Convert").id
            op = graph.add_node(f"l{layer}.op{i}", "MatMul").id
            graph.add_edge(weight, 0, convert, 0, ["64", "64"], "fp16")
            graph.add_edge(previous, 0, op, 0, ["?", "64"], "fp32")
            graph.add_edge(convert, 0, op, 1, ["64", "64"], "fp32")
            previous = op
        graph.add_edge(mask, 0, previous, 2, ["1", "?"], "bool")
    return graph


def test_passes_on_more_than_5000_nodes_are_fast():
    graph = build_large_graph()
    assert len(graph.nodes) > 5000
    pipeline = Pipeline([FoldConstantConvertsPass(), HideConstantsPass(), DetachFanoutPass(),
                         HighlightNodesPass(["l3.op7"])])
    start = time.perf_counter()
    result = pipeline.run(graph)
    elapsed = time.perf_counter() - start
    result.validate()
    assert len(result.of_type("Constant")) == 0
    assert len(result.of_type("Convert")) == 0
    # A linear-time implementation does this in well under a second; the limit is
    # generous so slow CI machines do not fail, but an O(n^2) bug would.
    assert elapsed < 5.0


def test_dot_generation_on_more_than_5000_nodes_is_fast():
    graph = build_large_graph()
    start = time.perf_counter()
    source = GraphvizRenderer().to_dot(graph).source
    assert time.perf_counter() - start < 5.0
    assert source.count(":s -> n") == len(graph.edges)   # one DOT edge line per graph edge
