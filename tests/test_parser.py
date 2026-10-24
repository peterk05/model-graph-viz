"""OpenVINO model parsing (brief section 4: 'Model parsing')."""

from pathlib import Path

import pytest

from mgv.graph import ROLE_CONSTANT, ROLE_INPUT, ROLE_OUTPUT
from mgv.parsers import ModelLoadError, load_model


def test_operation_types_and_roles_are_preserved(tiny_model_path):
    graph = load_model(tiny_model_path)
    graph.validate()
    types = graph.count_types()
    assert types["Parameter"] == 1
    assert types["MatMul"] == 1
    assert types["Split"] == 1
    assert types["Result"] == 2
    assert [n.name for n in graph.of_role(ROLE_INPUT)] == ["x"]
    assert len(graph.of_role(ROLE_OUTPUT)) == 2
    assert all(n.type == "Constant" for n in graph.of_role(ROLE_CONSTANT))


def test_connections_and_port_indices(tiny_model_path):
    graph = load_model(tiny_model_path)
    split = graph.nodes_by_name("split")[0]
    double = graph.nodes_by_name("double")[0]
    scaled = graph.nodes_by_name("scaled")[0]
    assert graph.port_counts(split.id)[1] == 2

    # Add(split.out0, split.out0): two parallel edges, both from output port 0.
    into_double = graph.inputs_of(double.id)
    assert [(e.src, e.src_port, e.dst_port) for e in into_double] == [(split.id, 0, 0), (split.id, 0, 1)]

    # split output 1 goes into input 0 of "scaled".
    into_scaled = graph.inputs_of(scaled.id)
    assert (into_scaled[0].src, into_scaled[0].src_port) == (split.id, 1)


def test_tensor_metadata_including_dynamic_dims(tiny_model_path):
    graph = load_model(tiny_model_path)
    matmul = graph.nodes_by_name("matmul")[0]
    data_in, weight_in = graph.inputs_of(matmul.id)
    assert data_in.shape == ["?", "4"]
    assert data_in.dtype == "fp32"
    assert weight_in.shape == ["4", "4"]
    assert weight_in.dtype == "fp32"   # after the Convert
    weight = graph.nodes_by_name("weight")[0]
    assert weight.role == ROLE_CONSTANT
    assert graph.outputs_of(weight.id)[0].dtype == "fp16"


def test_attributes_are_read_but_constant_values_are_not(tiny_model_path):
    graph = load_model(tiny_model_path)
    matmul = graph.nodes_by_name("matmul")[0]
    assert matmul.attrs["transpose_a"] == "false"
    weight = graph.nodes_by_name("weight")[0]
    assert set(weight.attrs) == {"element_type", "shape"}   # no tensor values copied


def test_missing_file_gives_clear_error(tmp_path):
    with pytest.raises(ModelLoadError, match="not found"):
        load_model(tmp_path / "missing.xml")


def test_unsupported_extension_gives_clear_error(tmp_path):
    path = tmp_path / "model.onnx"
    path.write_text("not really onnx")
    with pytest.raises(ModelLoadError, match="unsupported"):
        load_model(path)


def test_missing_weights_file_gives_clear_error(tiny_model_path: Path):
    tiny_model_path.with_suffix(".bin").unlink()
    with pytest.raises(ModelLoadError, match="weights file not found"):
        load_model(tiny_model_path)


def test_invalid_xml_gives_clear_error(tmp_path):
    path = tmp_path / "broken.xml"
    path.write_text("<net><this is not valid")
    path.with_suffix(".bin").write_bytes(b"")
    with pytest.raises(ModelLoadError, match="could not read"):
        load_model(path)
