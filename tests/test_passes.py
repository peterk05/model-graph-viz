"""Transformation passes on small synthetic graphs (no parsing, no rendering)."""

import pytest

from mgv.graph import VIZ_BORDER_COLOR, VIZ_INPUT_ANNOTATIONS, VIZ_NOTES, Graph
from mgv.passes import (
    DetachFanoutPass,
    FoldConstantConvertsPass,
    HideConstantsPass,
    HighlightNodesPass,
    PassConfigError,
    create_pass,
)

# ---------- highlight ----------


def test_highlight_marks_only_matching_nodes(branching_graph):
    result = HighlightNodesPass(["add", "mul"]).apply(branching_graph)
    highlighted = {n.name for n in result.nodes.values() if n.viz.get(VIZ_BORDER_COLOR) == "red"}
    assert highlighted == {"add", "mul"}
    untouched = [n for n in result.nodes.values() if n.name not in ("add", "mul")]
    assert all(VIZ_BORDER_COLOR not in n.viz for n in untouched)


def test_highlight_matches_every_node_with_a_duplicate_name():
    graph = Graph()
    graph.add_node("dup", "Add")
    graph.add_node("dup", "Add")
    graph.add_node("other", "Add")
    HighlightNodesPass(["dup"]).apply(graph)
    assert [VIZ_BORDER_COLOR in n.viz for n in graph.nodes.values()] == [True, True, False]


def test_highlight_reports_unmatched_names_and_custom_color(branching_graph):
    highlight = HighlightNodesPass(["add", "does_not_exist"], color="blue")
    highlight.apply(branching_graph)
    assert highlight.unmatched == {"does_not_exist"}
    assert branching_graph.nodes[2].viz[VIZ_BORDER_COLOR] == "blue"


def test_highlight_does_not_change_structure(branching_graph):
    before = (len(branching_graph.nodes), len(branching_graph.edges))
    HighlightNodesPass(["add"]).apply(branching_graph)
    assert (len(branching_graph.nodes), len(branching_graph.edges)) == before


# ---------- hide constants ----------


def test_hide_single_consumer_constant(matmul_graph):
    result = HideConstantsPass().apply(matmul_graph)
    result.validate()
    assert 1 not in result.nodes                        # constant removed
    matmul = result.nodes[2]
    assert matmul.viz[VIZ_INPUT_ANNOTATIONS] == {1: "fp32[1024, 768]"}   # original port 1 kept
    assert "shared" not in matmul.viz[VIZ_INPUT_ANNOTATIONS][1]
    # unrelated connections preserved
    assert [(e.src, e.dst_port) for e in result.inputs_of(2)] == [(0, 0)]
    assert [(e.src, e.dst) for e in result.outputs_of(2)] == [(2, 3)]


def test_hide_shared_constant_marks_every_consumer_with_its_id(branching_graph):
    result = HideConstantsPass().apply(branching_graph)
    result.validate()
    assert 4 not in result.nodes
    assert result.nodes[3].viz[VIZ_INPUT_ANNOTATIONS] == {1: "fp16[4, 4] (shared: 4)"}   # matmul
    assert result.nodes[5].viz[VIZ_INPUT_ANNOTATIONS] == {1: "fp16[4, 4] (shared: 4)"}   # mul


def test_hide_constants_preserves_unrelated_edges_including_parallel_ones(branching_graph):
    before = {(e.src, e.src_port, e.dst, e.dst_port) for e in branching_graph.edges.values() if e.src != 4}
    result = HideConstantsPass().apply(branching_graph)
    after = {(e.src, e.src_port, e.dst, e.dst_port) for e in result.edges.values()}
    assert after == before
    assert len(result.inputs_of(6)) == 2


def test_constant_stays_while_it_still_has_a_visible_connection():
    graph = Graph()
    c = graph.add_node("c", "Constant", role="constant")
    relu = graph.add_node("relu", "Relu")
    out = graph.add_node("out", "Result", role="output")
    graph.add_edge(c.id, 0, relu.id, 0, ["2"], "fp32")
    graph.add_edge(c.id, 0, out.id, 0, ["2"], "fp32")    # Result is kept visible by default
    result = HideConstantsPass().apply(graph)
    assert c.id in result.nodes
    assert [e.dst for e in result.outputs_of(c.id)] == [out.id]
    assert result.nodes[relu.id].viz[VIZ_INPUT_ANNOTATIONS] == {0: "fp32[2] (shared: 0)"}


def test_hide_constants_with_missing_metadata_and_unconnected_constant():
    graph = Graph()
    c = graph.add_node("c", "Constant", role="constant")
    lonely = graph.add_node("lonely", "Constant", role="constant")
    relu = graph.add_node("relu", "Relu")
    graph.add_edge(c.id, 0, relu.id, 0)                  # no shape or dtype known
    result = HideConstantsPass().apply(graph)
    assert result.nodes[relu.id].viz[VIZ_INPUT_ANNOTATIONS] == {0: "?[...]"}
    assert lonely.id not in result.nodes


def test_hide_constants_does_not_copy_values(matmul_graph):
    result = HideConstantsPass().apply(matmul_graph)
    annotation = result.nodes[2].viz[VIZ_INPUT_ANNOTATIONS][1]
    assert annotation == "fp32[1024, 768]"


# ---------- fold converts (custom) ----------


def _decompression_graph() -> Graph:
    """x -> MatMul <- Convert(fp16 -> fp32) <- Constant fp16 ; Constant also feeds a 2nd Convert."""
    graph = Graph()
    x = graph.add_node("x", "Parameter", role="input")
    w = graph.add_node("w", "Constant", role="constant")
    cv = graph.add_node("w/convert", "Convert", {"destination_type": "f32"})
    mm = graph.add_node("mm", "MatMul")
    act_cv = graph.add_node("act/convert", "Convert")   # converts an activation: must stay
    graph.add_edge(w.id, 0, cv.id, 0, ["4", "4"], "fp16")
    graph.add_edge(x.id, 0, mm.id, 0, ["?", "4"], "fp32")
    graph.add_edge(cv.id, 0, mm.id, 1, ["4", "4"], "fp32")
    graph.add_edge(mm.id, 0, act_cv.id, 0, ["?", "4"], "fp32")
    return graph


def test_fold_converts_connects_constant_directly_and_keeps_port():
    graph = FoldConstantConvertsPass().apply(_decompression_graph())
    graph.validate()
    assert graph.of_type("Convert") == [graph.nodes[4]]          # only the activation Convert left
    into_mm = graph.inputs_of(3)
    assert [(e.src, e.dst_port, e.dtype) for e in into_mm] == [(0, 0, "fp32"), (1, 1, "fp16→fp32")]


def test_fold_then_hide_gives_annotation_with_both_types():
    graph = _decompression_graph()
    for transformation_pass in (FoldConstantConvertsPass(), HideConstantsPass()):
        graph = transformation_pass.apply(graph)
    assert graph.nodes[3].viz[VIZ_INPUT_ANNOTATIONS] == {1: "fp16→fp32[4, 4]"}
    assert graph.count_types() == {"Parameter": 1, "MatMul": 1, "Convert": 1}


# ---------- detach fan-out (custom) ----------


def test_detach_fanout_replaces_long_edges_with_references():
    graph = Graph()
    mask = graph.add_node("mask", "Select")
    layers = [graph.add_node(f"layer{i}", "Add") for i in range(5)]
    for layer in layers:
        graph.add_edge(mask.id, 0, layer.id, 1, ["1", "?"], "bool")
    result = DetachFanoutPass(min_uses=3).apply(graph)
    result.validate()
    assert result.out_edges[mask.id] == []
    for layer in layers:
        (edge,) = result.inputs_of(layer.id)
        reference = result.nodes[edge.src]
        assert reference.role == "reference"
        assert reference.viz["reference_to"] == mask.id
        assert (edge.dst_port, edge.dtype, edge.shape) == (1, "bool", ["1", "?"])
    assert result.nodes[mask.id].viz[VIZ_NOTES] == [f"out 0: 5 uses shown as ↳ #{mask.id} references"]


def test_detach_fanout_leaves_small_fanout_alone(branching_graph):
    before = {(e.src, e.src_port, e.dst, e.dst_port) for e in branching_graph.edges.values()}
    result = DetachFanoutPass(min_uses=8).apply(branching_graph)
    assert {(e.src, e.src_port, e.dst, e.dst_port) for e in result.edges.values()} == before


# ---------- configuration ----------


def test_create_pass_from_string_options(tmp_path):
    names_file = tmp_path / "names.txt"
    names_file.write_text("b\n\nc\n")
    highlight = create_pass("highlight", {"names": "a", "names-file": str(names_file), "color": "green"})
    assert highlight.names == {"a", "b", "c"} and highlight.color == "green"
    assert create_pass("hide-constants", {"keep-for": "Result,Concat"}).keep_for == {"Result", "Concat"}
    assert create_pass("detach-fanout", {"min-uses": "4"}).min_uses == 4


@pytest.mark.parametrize(
    ("name", "options", "message"),
    [
        ("no-such-pass", {}, "unknown pass"),
        ("highlight", {}, "at least one node name"),
        ("highlight", {"colour": "red", "names": "a"}, "unknown option"),
        ("detach-fanout", {"min-uses": "lots"}, "whole number"),
        ("detach-fanout", {"min-uses": "1"}, "at least 2"),
    ],
)
def test_invalid_pass_configuration_is_reported(name, options, message):
    with pytest.raises(PassConfigError, match=message):
        create_pass(name, options)
