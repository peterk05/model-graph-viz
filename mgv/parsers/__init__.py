"""Model parsers. Importing this package registers every built-in parser."""

from mgv.parsers.base import (
    ModelLoadError,
    ModelParser,
    get_parser,
    load_model,
    register_parser,
    supported_extensions,
)
from mgv.parsers.openvino_parser import OpenVINOParser

__all__ = [
    "ModelLoadError",
    "ModelParser",
    "OpenVINOParser",
    "get_parser",
    "load_model",
    "register_parser",
    "supported_extensions",
]
