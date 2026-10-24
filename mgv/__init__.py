"""mgv: model graph visualization with transformation passes.

Pipeline: load a model -> framework-independent Graph -> passes -> SVG.

Quick start::

    from mgv import visualize, HideConstantsPass
    report = visualize("model.xml", "graph.svg", passes=[HideConstantsPass()])
    print(report.summary())
"""

from mgv.graph import Edge, Graph, GraphError, Node, format_tensor
from mgv.parsers import ModelLoadError, ModelParser, load_model, register_parser
from mgv.passes import (
    DetachFanoutPass,
    FoldConstantConvertsPass,
    HideConstantsPass,
    HighlightNodesPass,
    PassConfigError,
    TransformationPass,
    available_passes,
    create_pass,
    register_pass,
)
from mgv.pipeline import Pipeline, VisualizationReport, visualize
from mgv.render import GraphvizRenderer, Renderer, RenderError

__all__ = [
    "DetachFanoutPass",
    "Edge",
    "FoldConstantConvertsPass",
    "Graph",
    "GraphError",
    "GraphvizRenderer",
    "HideConstantsPass",
    "HighlightNodesPass",
    "ModelLoadError",
    "ModelParser",
    "Node",
    "PassConfigError",
    "Pipeline",
    "RenderError",
    "Renderer",
    "TransformationPass",
    "VisualizationReport",
    "available_passes",
    "create_pass",
    "format_tensor",
    "load_model",
    "register_parser",
    "register_pass",
    "visualize",
]
