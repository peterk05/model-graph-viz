"""Intermediate graph representation (brief section 4: 'Intermediate graph representation')."""

import pytest

from mgv.graph import ROLE_CONSTANT, Graph, GraphError, format_tensor


def test_nodes_get_unique_ids_separate_from_names():
    graph = Graph()
    first = graph.add_node("same_name", "Add")
    second = graph.add_node("same_name", "Add")
    assert first.id != second.id
    assert [node.id for node in graph.nodes_by_name("same_name")] == [first.id, second.id]


def test_edge_keeps_source_and_destination_ports(branching_graph):
    split_to_add = [e for e in branching_graph.inputs_of(2)]
    assert [(e.src, e.src_port, e.dst, e.dst_port) for e in split_to_add] == [(1, 0, 2, 0), (1, 1, 2, 1)]


def test_multiple_outputs_and_fan_out(branching_graph):
    split_outputs = branching_graph.outputs_of(1)
    assert sorted((e.src_port, e.dst, e.dst_port) for e in split_outputs) == [(0, 2, 0), (0, 3, 0), (1, 2, 1)]
    assert branching_graph.port_counts(1) == (1, 2)


def test_parallel_edges_between_same_nodes_are_kept(branching_graph):
    into_double = branching_graph.inputs_of(6)
    assert len(into_double) == 2
    assert {e.src for e in into_double} == {2}
    assert [e.dst_port for e in into_double] == [0, 1]
    assert into_double[0].id != into_double[1].id


def test_dynamic_shapes_and_missing_metadata():
    graph = Graph()
    a = graph.add_node("a", "Parameter")
    b = graph.add_node("b", "Relu")
    c = graph.add_node("c", "Relu")
    dynamic = graph.add_edge(a.id, 0, b.id, 0, ["?", "1..10", "64"], "fp32")
    unknown = graph.add_edge(b.id, 0, c.id, 0)          # no shape, no dtype
    assert dynamic.shape == ["?", "1..10", "64"]
    assert format_tensor(dynamic.dtype, dynamic.shape) == "fp32[?, 1..10, 64]"
    assert unknown.shape is None and unknown.dtype is None
    assert format_tensor(unknown.dtype, unknown.shape) == "?[...]"
    assert format_tensor("fp32", []) == "fp32[]"


def test_add_edge_rejects_unknown_nodes_and_negative_ports():
    graph = Graph()
    a = graph.add_node("a", "Relu")
    with pytest.raises(GraphError):
        graph.add_edge(a.id, 0, 99, 0)
    with pytest.raises(GraphError):
        graph.add_edge(a.id, -1, a.id, 0)


def test_unknown_role_is_rejected():
    with pytest.raises(GraphError):
        Graph().add_node("a", "Relu", role="banana")


def test_remove_node_also_removes_its_edges_and_graph_stays_valid(branching_graph):
    branching_graph.remove_node(3)   # matmul
    branching_graph.validate()
    assert 3 not in branching_graph.nodes
    assert all(e.src != 3 and e.dst != 3 for e in branching_graph.edges.values())
    # an unrelated parallel edge pair is untouched
    assert len(branching_graph.inputs_of(6)) == 2


def test_copy_is_independent(branching_graph):
    copy = branching_graph.copy()
    copy.nodes[2].viz["border_color"] = "red"
    copy.nodes[2].attrs["new"] = "x"
    copy.remove_node(4)
    assert "border_color" not in branching_graph.nodes[2].viz
    assert "new" not in branching_graph.nodes[2].attrs
    assert 4 in branching_graph.nodes
    branching_graph.validate()
    copy.validate()


def test_new_ids_after_copy_do_not_clash(branching_graph):
    copy = branching_graph.copy()
    new_node = copy.add_node("extra", "Relu")
    assert new_node.id not in branching_graph.nodes


def test_validate_detects_broken_index(branching_graph):
    branching_graph.in_edges[2].clear()   # corrupt the index on purpose
    with pytest.raises(GraphError):
        branching_graph.validate()


def test_roles_and_type_queries(matmul_graph):
    assert [n.name for n in matmul_graph.of_role(ROLE_CONSTANT)] == ["weight"]
    assert matmul_graph.count_types() == {"Parameter": 1, "Constant": 1, "MatMul": 1, "Result": 1}
    assert repr(matmul_graph) == "Graph(4 nodes, 3 edges)"
