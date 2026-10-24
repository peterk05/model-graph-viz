"""Graph rendering: valid SVG, ports, labels, and visualization metadata in the output."""

import xml.etree.ElementTree as ET

import pytest

from mgv.graph import Graph
from mgv.passes import DetachFanoutPass, HideConstantsPass, HighlightNodesPass
from mgv.pipeline import Pipeline
from mgv.render import GraphvizRenderer, RenderError

SVG_NS = "{http://www.w3.org/2000/svg}"


def _render(graph, tmp_path, **options):
    path = GraphvizRenderer(**options).render(graph, tmp_path / "out.svg")
    text = path.read_text(encoding="utf-8")
    root = ET.fromstring(text)          # raises if the SVG is not well-formed XML
    assert root.tag == f"{SVG_NS}svg"
    return text, root


def test_renders_valid_svg_with_every_node_and_edge(branching_graph, tmp_path):
    text, root = _render(branching_graph, tmp_path)
    groups = root.iter(f"{SVG_NS}g")
    classes = [g.get("class") for g in groups]
    assert classes.count("node") == len(branching_graph.nodes)
    assert classes.count("edge") == len(branching_graph.edges)
    for op_type in ("Split", "MatMul", "Multiply", "Parameter", "Result"):
        assert op_type in text


def test_ports_are_shown_and_edges_attach_to_them(branching_graph, tmp_path):
    dot = GraphvizRenderer().to_dot(branching_graph).source
    assert "n1:out1:s -> n2:in1:n" in dot      # Split out 1 -> Add in 1
    assert dot.count("n2:out0:s -> n6:in") == 2  # both parallel edges are drawn
    text, _ = _render(branching_graph, tmp_path)
    assert "out 1: fp32[?, 4]" in text           # output port with its tensor type
    assert "in 1" in text


def test_attributes_are_displayed(branching_graph, tmp_path):
    text, _ = _render(branching_graph, tmp_path)
    assert "num_splits = 2" in text


def test_highlight_is_reflected_in_svg(branching_graph, tmp_path):
    result = Pipeline([HighlightNodesPass(["matmul"])]).run(branching_graph)
    text, root = _render(result, tmp_path)
    red_groups = []
    for group in root.iter(f"{SVG_NS}g"):
        if group.get("class") != "node":
            continue
        title = group.find(f"{SVG_NS}title").text
        strokes = {el.get("stroke") for el in group.iter() if el.get("stroke")}
        if "red" in strokes:
            red_groups.append(title)
    assert red_groups == ["n3"]                 # only the MatMul node has a red border


def test_hidden_constant_annotations_are_displayed(matmul_graph, tmp_path):
    result = Pipeline([HideConstantsPass()]).run(matmul_graph)
    text, _ = _render(result, tmp_path)
    assert "[1]: fp32[1024, 768]" in text
    assert "Constant" not in text


def test_reference_nodes_and_notes_render(tmp_path):
    graph = Graph()
    source = graph.add_node("mask", "Select")
    for i in range(3):
        graph.add_edge(source.id, 0, graph.add_node(f"use{i}", "Add").id, 0, ["1"], "bool")
    result = Pipeline([DetachFanoutPass(min_uses=2)]).run(graph)
    text, _ = _render(result, tmp_path)
    assert text.count("↳ Select #0 out 0") == 3
    assert "3 uses shown as" in text


def test_unknown_operation_types_render(tmp_path):
    graph = Graph()
    a = graph.add_node("a", "MyBrandNewOp", {"weird & <attr>": "x > y"})
    b = graph.add_node("b", "AnotherCustomOp")
    graph.add_edge(a.id, 3, b.id, 2, None, None)     # high port numbers, no metadata
    text, _ = _render(graph, tmp_path)
    assert "MyBrandNewOp" in text and "AnotherCustomOp" in text
    assert "weird &amp; &lt;attr&gt; = x &gt; y" in text   # special characters escaped


def test_optional_edge_labels(branching_graph, tmp_path):
    with_labels = GraphvizRenderer(show_edge_labels=True).to_dot(branching_graph).source
    without = GraphvizRenderer().to_dot(branching_graph).source
    assert 'label="fp16[4, 4]"' in with_labels
    assert 'label="fp16[4, 4]"' not in without


def test_large_graph_gets_fast_layout_setting():
    graph = Graph()
    for i in range(10):
        graph.add_node(f"n{i}", "Relu")
    assert "nslimit" in GraphvizRenderer(large_graph_threshold=5).to_dot(graph).source
    assert "nslimit" not in GraphvizRenderer(large_graph_threshold=50).to_dot(graph).source


def test_wrong_output_extension_is_rejected(matmul_graph, tmp_path):
    with pytest.raises(RenderError, match=".svg"):
        GraphvizRenderer().render(matmul_graph, tmp_path / "out.png")


def test_save_dot_writes_source(matmul_graph, tmp_path):
    GraphvizRenderer(save_dot=True).render(matmul_graph, tmp_path / "g.svg")
    assert (tmp_path / "g.dot").read_text().startswith("digraph")
