"""Shared test fixtures: small synthetic graphs and a tiny OpenVINO model saved to disk."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make `import mgv` work when running `pytest` from the repository root without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mgv.graph import ROLE_CONSTANT, ROLE_INPUT, ROLE_OUTPUT, Graph  # noqa: E402


@pytest.fixture
def matmul_graph() -> Graph:
    """The brief's example: Input -> MatMul <- Constant fp32[1024, 768] -> Output.

    ids: 0 input, 1 weight, 2 matmul, 3 output
    """
    graph = Graph("matmul")
    x = graph.add_node("input", "Parameter", role=ROLE_INPUT)
    w = graph.add_node("weight", "Constant", {"element_type": "fp32", "shape": "[1024, 768]"},
                       role=ROLE_CONSTANT)
    mm = graph.add_node("matmul", "MatMul", {"transpose_b": "false"})
    out = graph.add_node("output", "Result", role=ROLE_OUTPUT)
    graph.add_edge(x.id, 0, mm.id, 0, ["1", "1024"], "fp32")
    graph.add_edge(w.id, 0, mm.id, 1, ["1024", "768"], "fp32")
    graph.add_edge(mm.id, 0, out.id, 0, ["1", "768"], "fp32")
    return graph


@pytest.fixture
def branching_graph() -> Graph:
    """The brief's port example (Split -> Add / MatMul) plus a shared constant.

        x ---------------> Split            (one input, two outputs)
        Split out 0 -----> Add in 0
        Split out 1 -----> Add in 1
        Split out 0 -----> MatMul in 0      (the same output port feeds two consumers)
        shared_w out 0 --> MatMul in 1      (a constant with two consumers ...)
        shared_w out 0 --> Mul in 1         (... so it is "shared")
        MatMul out 0 ----> Mul in 0
        Add out 0 -------> double in 0      (two parallel edges between
        Add out 0 -------> double in 1       the same pair of nodes)
        double out 0 ----> out

    ids: 0 x, 1 split, 2 add, 3 matmul, 4 shared_w, 5 mul, 6 double, 7 out
    """
    graph = Graph("branching")
    x = graph.add_node("x", "Parameter", role=ROLE_INPUT)
    split = graph.add_node("split", "Split", {"num_splits": "2"}, num_inputs=1, num_outputs=2)
    add = graph.add_node("add", "Add")
    matmul = graph.add_node("matmul", "MatMul")
    shared = graph.add_node("shared_w", "Constant", role=ROLE_CONSTANT)
    mul = graph.add_node("mul", "Multiply")
    double = graph.add_node("double", "Add")
    out = graph.add_node("out", "Result", role=ROLE_OUTPUT)

    graph.add_edge(x.id, 0, split.id, 0, ["?", "8"], "fp32")
    graph.add_edge(split.id, 0, add.id, 0, ["?", "4"], "fp32")
    graph.add_edge(split.id, 1, add.id, 1, ["?", "4"], "fp32")
    graph.add_edge(split.id, 0, matmul.id, 0, ["?", "4"], "fp32")
    graph.add_edge(shared.id, 0, matmul.id, 1, ["4", "4"], "fp16")
    graph.add_edge(shared.id, 0, mul.id, 1, ["4", "4"], "fp16")
    graph.add_edge(matmul.id, 0, mul.id, 0, ["?", "4"], "fp32")
    graph.add_edge(add.id, 0, double.id, 0, ["?", "4"], "fp32")
    graph.add_edge(add.id, 0, double.id, 1, ["?", "4"], "fp32")
    graph.add_edge(double.id, 0, out.id, 0, ["?", "4"], "fp32")
    return graph


def build_tiny_openvino_model(directory: Path) -> Path:
    """Save a tiny but varied OpenVINO IR model and return the .xml path.

    Parameter x [?, 4] -> MatMul(x, Convert(fp16 const)) -> Split into 2
    -> Add(out0, out0) -> Result;  Split out1 -> Result
    The fp16 constant is also used by a second Convert -> Multiply (shared constant).
    """
    import numpy as np
    import openvino as ov
    import openvino.opset13 as ops

    x = ops.parameter(ov.PartialShape([-1, 4]), ov.Type.f32, name="x")
    weight = ops.constant(np.ones((4, 4), dtype=np.float16))
    weight.set_friendly_name("weight")
    matmul = ops.matmul(x, ops.convert(weight, ov.Type.f32), False, False)
    matmul.set_friendly_name("matmul")
    split = ops.split(matmul, ops.constant(np.int64(1)), 2)
    split.set_friendly_name("split")
    double = ops.add(split.output(0), split.output(0))
    double.set_friendly_name("double")
    scaled = ops.multiply(split.output(1), ops.convert(ops.slice(weight, np.array([0, 0]), np.array([1, 2]),
                                                                    np.array([1, 1])), ov.Type.f32))
    scaled.set_friendly_name("scaled")
    model = ov.Model([ops.result(double), ops.result(scaled)], [x], "tiny")

    path = directory / "tiny.xml"
    ov.save_model(model, path, compress_to_fp16=False)
    return path


@pytest.fixture
def tiny_model_path(tmp_path: Path) -> Path:
    return build_tiny_openvino_model(tmp_path)
