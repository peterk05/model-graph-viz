"""Generate every example SVG in examples/ and the benchmark table in the README.

For each model it renders the same graph with different pass combinations, times
every stage, writes examples/benchmark.md, and copies that table into README.md
between the BENCHMARK markers.

Usage (from the repository root):
    python scripts/build_example_models.py      # small models (already committed)
    bash scripts/export_tinyllama.sh            # TinyLlama, ~2.2 GB, not committed
    python scripts/make_examples.py             # all examples + benchmark (~5 min)
    python scripts/make_examples.py --skip-tinyllama
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mgv import (  # noqa: E402
    DetachFanoutPass,
    FoldConstantConvertsPass,
    HideConstantsPass,
    HighlightNodesPass,
    TransformationPass,
    VisualizationReport,
    visualize,
)

EXAMPLES = REPO_ROOT / "examples"
README = REPO_ROOT / "README.md"
BENCH_START = "<!-- BENCHMARK:START -->"
BENCH_END = "<!-- BENCHMARK:END -->"


@dataclass
class ModelExample:
    name: str
    path: Path
    highlight: list[str]
    custom: list[str]  # names of the custom passes worth showing for this model


TINYLLAMA_HIGHLIGHT = [
    "__module.model.layers.0.self_attn/aten::scaled_dot_product_attention/ScaledDotProductAttention",
    "__module.model.layers.0.mlp.down_proj/ov_ext::linear/MatMul",
    "__module.lm_head/ov_ext::linear/MatMul",
]

MODELS = [
    ModelExample("small_cnn", EXAMPLES / "models/small_cnn/small_cnn.xml",
                 ["split_channels", "merge"], []),
    ModelExample("mini_attention", EXAMPLES / "models/mini_attention/mini_attention.xml",
                 ["attention", "residual"], ["fold-converts"]),
    ModelExample("tinyllama", REPO_ROOT / "models/tinyllama/openvino_model.xml",
                 TINYLLAMA_HIGHLIGHT, ["fold-converts", "detach-fanout"]),
]


def variants(example: ModelExample) -> list[tuple[str, list[TransformationPass]]]:
    """(file name, passes) for every SVG we make of this model."""
    result: list[tuple[str, list[TransformationPass]]] = [
        ("1_original.svg", []),
        ("2_hide_constants.svg", [HideConstantsPass()]),
        ("3_highlight.svg", [HideConstantsPass(), HighlightNodesPass(example.highlight)]),
    ]
    if example.custom:
        custom: list[TransformationPass] = []
        if "fold-converts" in example.custom:
            custom.append(FoldConstantConvertsPass())
        custom.append(HideConstantsPass())
        if "detach-fanout" in example.custom:
            custom.append(DetachFanoutPass())
        custom.append(HighlightNodesPass(example.highlight))
        result.append(("4_custom_passes.svg", custom))
    return result


def environment() -> str:
    cpu = platform.processor() or "unknown CPU"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    memory = "unknown"
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                memory = f"{int(line.split()[1]) / 1024 / 1024:.1f} GB"
                break
    except OSError:
        pass
    try:
        dot_version = subprocess.run(["dot", "-V"], capture_output=True, text=True, check=False).stderr.strip()
    except OSError:
        dot_version = "dot not found"
    import graphviz
    import openvino

    return (f"{platform.system()} {platform.release()} | CPU: {cpu} ({os.cpu_count()} threads) | "
            f"RAM: {memory} | Python {platform.python_version()} | OpenVINO {openvino.__version__} | "
            f"{dot_version} | graphviz (Python) {graphviz.__version__}")


def benchmark_table(rows: list[tuple[str, str, VisualizationReport]]) -> str:
    lines = [
        "| Model | Passes | Nodes / edges before | Nodes / edges after "
        "| Load + build graph (s) | Passes (s) | Render SVG (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for model, passes, report in rows:
        lines.append(
            f"| {model} | {passes or '(none)'} | {report.nodes_before} / {report.edges_before} | "
            f"{report.nodes_after} / {report.edges_after} | {report.load_seconds:.2f} | "
            f"{report.transform_seconds:.3f} | {report.render_seconds:.2f} |"
        )
    return "\n".join(lines)


def update_readme(table: str, env: str) -> None:
    if not README.exists():
        return
    text = README.read_text(encoding="utf-8")
    if BENCH_START not in text or BENCH_END not in text:
        return
    before = text.split(BENCH_START)[0]
    after = text.split(BENCH_END)[1]
    block = f"{BENCH_START}\n**Test environment:** {env}\n\n{table}\n{BENCH_END}"
    README.write_text(before + block + after, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-tinyllama", action="store_true")
    args = parser.parse_args()

    rows: list[tuple[str, str, VisualizationReport]] = []
    failures = 0
    for example in MODELS:
        if example.name == "tinyllama" and args.skip_tinyllama:
            continue
        if not example.path.exists():
            print(f"skipping {example.name}: {example.path} not found "
                  f"(see scripts/export_tinyllama.sh)", file=sys.stderr)
            continue
        for file_name, passes in variants(example):
            output = EXAMPLES / example.name / file_name
            pass_names = ",".join(p.name for p in passes)
            print(f"{example.name}: {file_name} [{pass_names or 'no passes'}] ...", flush=True)
            try:
                report = visualize(example.path, output, passes)
            except Exception as error:   # keep going so one failure doesn't lose every other example
                print(f"FAILED {example.name} {file_name}: {error}", file=sys.stderr)
                failures += 1
                continue
            print(report.summary(), "\n", flush=True)
            rows.append((example.name, pass_names, report))

    env = environment()
    table = benchmark_table(rows)
    (EXAMPLES / "benchmark.md").write_text(f"# Benchmark\n\n**Test environment:** {env}\n\n{table}\n",
                                           encoding="utf-8")
    update_readme(table, env)
    print(f"wrote {EXAMPLES / 'benchmark.md'} and updated README.md")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
