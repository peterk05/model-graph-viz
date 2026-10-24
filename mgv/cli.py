"""Command-line interface, built only on the public Python API.

Examples
--------
    python model_graph_viewer.py --model model.xml --output graph.svg
    python model_graph_viewer.py --model model.xml --output simple.svg \\
        --passes fold-converts,hide-constants,highlight \\
        --pass-option highlight.names=MatMul_1,Add_7
    python model_graph_viewer.py --list-passes
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from mgv.parsers import ModelLoadError, supported_extensions
from mgv.passes import PassConfigError, TransformationPass, available_passes, create_pass
from mgv.passes.base import split_list
from mgv.passes.highlight import HighlightNodesPass
from mgv.pipeline import visualize
from mgv.render import GraphvizRenderer, RenderError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model_graph_viewer.py",
        description="Visualize a neural network graph as an SVG, optionally simplified by "
                    "transformation passes applied in the order given.",
    )
    parser.add_argument("--model", help=f"model file to load (supported: {', '.join(supported_extensions())})")
    parser.add_argument("--output", help="SVG file to write")
    parser.add_argument(
        "--passes", default="",
        help="comma-separated pass names, applied in this order (default: none). "
             "See --list-passes.",
    )
    parser.add_argument(
        "--pass-option", action="append", default=[], metavar="PASS.KEY=VALUE",
        help="configure a pass, e.g. highlight.names=MatMul_1,Add_7 (repeatable)",
    )
    parser.add_argument("--edge-labels", action="store_true",
                        help="also print dtype/shape on every edge (fine for small graphs; "
                             "much slower layout for large ones)")
    parser.add_argument("--save-dot", action="store_true",
                        help="also save the Graphviz .dot source next to the SVG")
    parser.add_argument("--list-passes", action="store_true", help="list available passes and exit")
    return parser


def parse_pass_options(raw_options: list[str]) -> dict[str, dict[str, str]]:
    """['highlight.names=a,b', 'highlight.color=blue'] -> {'highlight': {'names': 'a,b', 'color': 'blue'}}"""
    options: dict[str, dict[str, str]] = {}
    for raw in raw_options:
        if "=" not in raw or "." not in raw.split("=", 1)[0]:
            raise PassConfigError(f"invalid --pass-option {raw!r}: expected PASS.KEY=VALUE")
        target, value = raw.split("=", 1)
        pass_name, key = target.split(".", 1)
        if not pass_name or not key:
            raise PassConfigError(f"invalid --pass-option {raw!r}: expected PASS.KEY=VALUE")
        options.setdefault(pass_name, {})[key] = value
    return options


def build_passes(pass_list: str, raw_options: list[str]) -> list[TransformationPass]:
    names = split_list(pass_list)
    options = parse_pass_options(raw_options)
    unused = sorted(set(options) - set(names))
    if unused:
        raise PassConfigError(f"options given for pass(es) not selected in --passes: {', '.join(unused)}")
    return [create_pass(name, options.get(name, {})) for name in names]


def list_passes() -> str:
    lines = ["Available passes:"]
    for name, pass_cls in available_passes().items():
        lines.append(f"  {name:<16} {pass_cls.description}")
        for option, help_text in pass_cls.option_help.items():
            lines.append(f"  {'':<16}   {name}.{option}=...  {help_text}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_passes:
        print(list_passes())
        return 0
    if not args.model or not args.output:
        parser.error("--model and --output are required (or use --list-passes)")

    try:
        passes = build_passes(args.passes, args.pass_option)
        renderer = GraphvizRenderer(show_edge_labels=args.edge_labels, save_dot=args.save_dot)
        report = visualize(args.model, args.output, passes, renderer)
    except (ModelLoadError, PassConfigError, RenderError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    for transformation_pass in passes:
        if isinstance(transformation_pass, HighlightNodesPass) and transformation_pass.unmatched:
            missing = ", ".join(sorted(transformation_pass.unmatched))
            print(f"warning: highlight: no node named {missing}", file=sys.stderr)
    print(report.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
