"""Running passes in order, and the one-call Python API ``visualize``.

    graph = load_model("model.xml")                       # 1-2. load + build graph
    result = Pipeline([FoldConstantConvertsPass(),
                       HideConstantsPass()]).run(graph)   # 3. transform (graph is untouched)
    GraphvizRenderer().render(result, Path("out.svg"))    # 4. render

``visualize`` does all four steps and reports how long each one took.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from mgv.graph import Graph
from mgv.parsers import load_model
from mgv.passes.base import TransformationPass
from mgv.render.base import Renderer
from mgv.render.graphviz_renderer import GraphvizRenderer


class Pipeline:
    """An ordered list of passes."""

    def __init__(self, passes: list[TransformationPass] | None = None) -> None:
        self.passes = list(passes) if passes is not None else []
        self.pass_seconds: list[tuple[str, float]] = []   # (pass name, seconds) from the last run

    def run(self, graph: Graph, copy: bool = True) -> Graph:
        """Apply every pass in order and return the result.

        With ``copy=True`` (the default) the input graph is copied once first, so the
        caller's graph is never changed. Each pass then edits that single copy.
        """
        result = graph.copy() if copy else graph
        self.pass_seconds = []
        for transformation_pass in self.passes:
            start = time.perf_counter()
            result = transformation_pass.apply(result)
            self.pass_seconds.append((transformation_pass.name, time.perf_counter() - start))
        return result


@dataclass
class VisualizationReport:
    """What ``visualize`` did: graph sizes and time spent in each stage (seconds)."""

    model_path: Path
    output_path: Path
    nodes_before: int
    edges_before: int
    nodes_after: int
    edges_after: int
    load_seconds: float
    pass_seconds: list[tuple[str, float]] = field(default_factory=list)
    render_seconds: float = 0.0

    @property
    def transform_seconds(self) -> float:
        return sum(seconds for _, seconds in self.pass_seconds)

    def summary(self) -> str:
        lines = [
            f"model:   {self.model_path}",
            f"graph:   {self.nodes_before} nodes, {self.edges_before} edges"
            f" -> {self.nodes_after} nodes, {self.edges_after} edges after passes",
            f"load + graph construction: {self.load_seconds:8.3f} s",
            f"transformation passes:     {self.transform_seconds:8.3f} s",
        ]
        for name, seconds in self.pass_seconds:
            lines.append(f"    {name:<22} {seconds:8.3f} s")
        lines.append(f"rendering to SVG:          {self.render_seconds:8.3f} s")
        lines.append(f"output:  {self.output_path}")
        return "\n".join(lines)


def visualize(
    model_path: str | Path,
    output_path: str | Path,
    passes: list[TransformationPass] | None = None,
    renderer: Renderer | None = None,
) -> VisualizationReport:
    """Load a model, apply passes in order, write an SVG, and time every stage."""
    model_path = Path(model_path)
    output_path = Path(output_path)
    renderer = renderer if renderer is not None else GraphvizRenderer()

    start = time.perf_counter()
    graph = load_model(model_path)
    load_seconds = time.perf_counter() - start

    nodes_before, edges_before = len(graph.nodes), len(graph.edges)
    pipeline = Pipeline(passes)
    # The freshly parsed graph is not used anywhere else, so the passes can edit it
    # directly: no copy needed (this saves memory on large models).
    result = pipeline.run(graph, copy=False)

    start = time.perf_counter()
    renderer.render(result, output_path)
    render_seconds = time.perf_counter() - start

    return VisualizationReport(
        model_path=model_path,
        output_path=output_path,
        nodes_before=nodes_before,
        edges_before=edges_before,
        nodes_after=len(result.nodes),
        edges_after=len(result.edges),
        load_seconds=load_seconds,
        pass_seconds=pipeline.pass_seconds,
        render_seconds=render_seconds,
    )
