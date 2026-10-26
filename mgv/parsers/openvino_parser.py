"""OpenVINO IR (.xml + .bin) -> Graph.

This is the only module in the project that imports ``openvino``. Everything it
produces is plain Python data (strings, ints, lists), so passes and renderers never
touch OpenVINO objects and the original model is never modified.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mgv.graph import ROLE_CONSTANT, ROLE_INPUT, ROLE_OP, ROLE_OUTPUT, Graph
from mgv.parsers.base import ModelLoadError, ModelParser, register_parser

# OpenVINO op types that map to framework-independent roles.
_ROLE_BY_TYPE = {
    "Parameter": ROLE_INPUT,
    "Result": ROLE_OUTPUT,
    "Constant": ROLE_CONSTANT,
}

# OpenVINO element type names -> the names used in the graph and in labels (e.g. "fp32").
_DTYPE_NAMES = {
    "f64": "fp64", "f32": "fp32", "f16": "fp16", "bf16": "bf16",
    "f8e4m3": "fp8e4m3", "f8e5m2": "fp8e5m2", "nf4": "nf4",
    "i64": "int64", "i32": "int32", "i16": "int16", "i8": "int8", "i4": "int4",
    "u64": "uint64", "u32": "uint32", "u16": "uint16", "u8": "uint8", "u4": "uint4", "u1": "uint1",
    "boolean": "bool", "string": "string",
}


@register_parser
class OpenVINOParser(ModelParser):
    extensions = (".xml",)
    format_name = "OpenVINO IR"

    def parse(self, path: Path) -> Graph:
        model = self._read_model(path)
        graph = Graph(name=model.get_friendly_name() or path.stem)

        # get_ordered_ops() returns every op with producers before consumers, so by the
        # time we look at an op's inputs, every source op already has a graph id.
        ops = model.get_ordered_ops()

        # op.get_name() is unique inside a model (friendly names are not), so it is a
        # safe key for "which graph node did this OpenVINO op become?".
        node_id_of: dict[str, int] = {}
        for op in ops:
            op_type = op.get_type_name()
            node = graph.add_node(
                name=op.get_friendly_name(),
                op_type=op_type,
                attrs=_read_attributes(op),
                role=_ROLE_BY_TYPE.get(op_type, ROLE_OP),
                num_inputs=op.get_input_size(),
                num_outputs=op.get_output_size(),
            )
            node_id_of[op.get_name()] = node.id

        for op in ops:
            dst = node_id_of[op.get_name()]
            for dst_port in range(op.get_input_size()):
                source_output = op.input(dst_port).get_source_output()
                src = node_id_of[source_output.get_node().get_name()]
                graph.add_edge(
                    src=src,
                    src_port=source_output.get_index(),
                    dst=dst,
                    dst_port=dst_port,
                    shape=_read_shape(source_output),
                    dtype=_read_dtype(source_output),
                )
        return graph

    @staticmethod
    def _read_model(path: Path) -> Any:
        if path.suffix.lower() != ".xml":
            raise ModelLoadError(f"OpenVINO IR models must be .xml files, got {str(path)!r}")
        weights = path.with_suffix(".bin")
        if not weights.exists():
            raise ModelLoadError(
                f"weights file not found: {str(weights)!r} (an OpenVINO IR model needs the .bin "
                f"file next to the .xml file)"
            )
        # Every IR file has a <net ...> root element near the top. Checking for it first gives
        # a clear message for other .xml files instead of OpenVINO's generic one.
        with open(path, encoding="utf-8", errors="replace") as file:
            if "<net" not in file.read(4096):
                raise ModelLoadError(f"{str(path)!r} is not an OpenVINO IR file (no <net> element)")
        try:
            import openvino as ov
        except ImportError as error:  # pragma: no cover - depends on the environment
            raise ModelLoadError("the 'openvino' package is not installed: pip install openvino") from error
        try:
            return ov.Core().read_model(str(path))
        except Exception as error:  # OpenVINO raises RuntimeError with a long C++ message
            raise ModelLoadError(f"OpenVINO could not read {str(path)!r}: {_reason(error)}") from error


def _reason(error: Exception) -> str:
    """The useful line of an OpenVINO error.

    OpenVINO messages start with lines like "Exception from src/.../core.cpp:84:" and
    "Check '...' failed at ...", which say where in C++ it failed, not why. The last
    other line is the reason, e.g. "Start-end tags mismatch at offset 299".
    """
    lines = [line.strip() for line in str(error).splitlines() if line.strip()]
    reasons = [line for line in lines if not line.startswith(("Exception from", "Check '"))]
    if reasons:
        return reasons[-1]
    return lines[-1] if lines else type(error).__name__


def _read_attributes(op: Any) -> dict[str, str]:
    """Operation attributes as strings. Constants report only their type and shape, never values."""
    try:
        raw = op.get_attributes()
    except Exception:  # a few op types cannot report attributes; show none rather than fail
        return {}
    return {str(key): _attribute_to_text(value) for key, value in raw.items()}


def _attribute_to_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_attribute_to_text(item) for item in value) + "]"
    return str(value)


def _read_shape(output: Any) -> list[str] | None:
    """Shape of a tensor as strings: '?' for a fully dynamic dim, '1..10' for a range.

    Returns None when even the number of dimensions (the rank) is unknown.
    """
    partial_shape = output.get_partial_shape()
    if partial_shape.rank.is_dynamic:
        return None
    dims: list[str] = []
    for dim in partial_shape:
        if dim.is_static:
            dims.append(str(dim.get_length()))
        elif dim.get_min_length() == 0 and dim.get_max_length() == -1:
            dims.append("?")
        else:
            dims.append(str(dim))   # an interval such as "1..10"
    return dims


def _read_dtype(output: Any) -> str | None:
    name = output.get_element_type().get_type_name()
    if name in ("dynamic", "undefined"):
        return None
    return _DTYPE_NAMES.get(name, name)
