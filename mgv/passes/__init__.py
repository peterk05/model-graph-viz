"""Transformation passes. Importing this package registers every built-in pass."""

from mgv.passes.base import (
    PassConfigError,
    TransformationPass,
    available_passes,
    create_pass,
    register_pass,
)
from mgv.passes.detach_fanout import DetachFanoutPass
from mgv.passes.fold_converts import FoldConstantConvertsPass
from mgv.passes.hide_constants import HideConstantsPass
from mgv.passes.highlight import HighlightNodesPass

__all__ = [
    "DetachFanoutPass",
    "FoldConstantConvertsPass",
    "HideConstantsPass",
    "HighlightNodesPass",
    "PassConfigError",
    "TransformationPass",
    "available_passes",
    "create_pass",
    "register_pass",
]
