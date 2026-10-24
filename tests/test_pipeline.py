"""Transformation pipeline: order, validity, and not touching the input graph."""

from mgv.graph import VIZ_BORDER_COLOR, VIZ_INPUT_ANNOTATIONS, Graph
from mgv.passes import HideConstantsPass, HighlightNodesPass, TransformationPass
from mgv.pipeline import Pipeline


class RecordingPass(TransformationPass):
    """Test helper: appends its label to a shared list when it runs."""

    name = "recording"

    def __init__(self, label: str, log: list[str]) -> None:
        self.label = label
        self.log = log

    def apply(self, graph: Graph) -> Graph:
        self.log.append(self.label)
        return graph


def test_passes_run_in_the_given_order(branching_graph):
    log: list[str] = []
    Pipeline([RecordingPass("first", log), RecordingPass("second", log), RecordingPass("third", log)]).run(
        branching_graph
    )
    assert log == ["first", "second", "third"]


def test_pipeline_does_not_modify_the_input_graph(branching_graph):
    before_nodes = set(branching_graph.nodes)
    before_edges = {(e.src, e.src_port, e.dst, e.dst_port) for e in branching_graph.edges.values()}
    result = Pipeline([HideConstantsPass(), HighlightNodesPass(["add"])]).run(branching_graph)
    assert set(branching_graph.nodes) == before_nodes
    assert {(e.src, e.src_port, e.dst, e.dst_port) for e in branching_graph.edges.values()} == before_edges
    assert all(not node.viz for node in branching_graph.nodes.values())
    assert result is not branching_graph


def test_combined_passes_give_a_valid_graph_with_both_effects(branching_graph):
    result = Pipeline([HideConstantsPass(), HighlightNodesPass(["matmul"])]).run(branching_graph)
    result.validate()
    matmul = result.nodes[3]
    assert matmul.viz[VIZ_BORDER_COLOR] == "red"
    assert matmul.viz[VIZ_INPUT_ANNOTATIONS] == {1: "fp16[4, 4] (shared: 4)"}


def test_order_matters_and_is_respected(matmul_graph):
    # Highlighting the constant before hiding it means the highlight disappears with it.
    first = Pipeline([HighlightNodesPass(["weight"]), HideConstantsPass()]).run(matmul_graph)
    assert 1 not in first.nodes
    # Hiding first means there is nothing named "weight" left to highlight.
    highlight = HighlightNodesPass(["weight"])
    Pipeline([HideConstantsPass(), highlight]).run(matmul_graph)
    assert highlight.unmatched == {"weight"}


def test_empty_pipeline_returns_an_equal_copy(matmul_graph):
    result = Pipeline([]).run(matmul_graph)
    assert result is not matmul_graph
    assert repr(result) == repr(matmul_graph)


def test_pipeline_records_time_per_pass(matmul_graph):
    pipeline = Pipeline([HideConstantsPass(), HighlightNodesPass(["matmul"])])
    pipeline.run(matmul_graph)
    assert [name for name, _ in pipeline.pass_seconds] == ["hide-constants", "highlight"]
    assert all(seconds >= 0 for _, seconds in pipeline.pass_seconds)
