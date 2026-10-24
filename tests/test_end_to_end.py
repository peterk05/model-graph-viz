"""End-to-end: a real (tiny) OpenVINO model through the Python API and the CLI."""

import subprocess
import sys
from pathlib import Path

from mgv import FoldConstantConvertsPass, HideConstantsPass, HighlightNodesPass, visualize
from mgv.cli import main

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_python_api_end_to_end(tiny_model_path, tmp_path):
    output = tmp_path / "tiny.svg"
    report = visualize(
        tiny_model_path,
        output,
        passes=[FoldConstantConvertsPass(), HideConstantsPass(), HighlightNodesPass(["split"])],
    )
    text = output.read_text(encoding="utf-8")
    assert text.lstrip().startswith("<?xml") and "<svg" in text
    assert "fp16→fp32[4, 4]" in text           # folded + hidden weight annotation
    assert 'stroke="red"' in text                # highlight
    assert report.nodes_after < report.nodes_before
    assert [name for name, _ in report.pass_seconds] == ["fold-converts", "hide-constants", "highlight"]
    assert report.load_seconds > 0 and report.render_seconds > 0
    assert "load + graph construction" in report.summary()


def test_cli_without_passes(tiny_model_path, tmp_path, capsys):
    output = tmp_path / "plain.svg"
    assert main(["--model", str(tiny_model_path), "--output", str(output)]) == 0
    assert output.exists()
    assert "rendering to SVG" in capsys.readouterr().out


def test_cli_with_passes_and_options(tiny_model_path, tmp_path, capsys):
    output = tmp_path / "simplified.svg"
    code = main([
        "--model", str(tiny_model_path), "--output", str(output),
        "--passes", "fold-converts,hide-constants,highlight",
        "--pass-option", "highlight.names=split,missing_name",
        "--pass-option", "highlight.color=blue",
    ])
    assert code == 0
    text = output.read_text(encoding="utf-8")
    assert 'stroke="blue"' in text
    captured = capsys.readouterr()
    assert "no node named missing_name" in captured.err


def test_cli_errors_are_clear(tmp_path, capsys):
    assert main(["--model", str(tmp_path / "missing.xml"), "--output", str(tmp_path / "x.svg")]) == 1
    assert "model file not found" in capsys.readouterr().err
    assert main(["--model", "a.xml", "--output", "x.svg", "--passes", "nope"]) == 1
    assert "unknown pass 'nope'" in capsys.readouterr().err
    assert main(["--model", "a.xml", "--output", "x.svg", "--pass-option", "highlight.names=a"]) == 1
    assert "not selected in --passes" in capsys.readouterr().err
    assert main(["--model", "a.xml", "--output", "x.svg", "--passes", "highlight",
                 "--pass-option", "bad-format"]) == 1
    assert "expected PASS.KEY=VALUE" in capsys.readouterr().err


def test_cli_list_passes(capsys):
    assert main(["--list-passes"]) == 0
    out = capsys.readouterr().out
    for name in ("highlight", "hide-constants", "fold-converts", "detach-fanout"):
        assert name in out


def test_entry_script_runs(tiny_model_path, tmp_path):
    output = tmp_path / "script.svg"
    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / "model_graph_viewer.py"),
         "--model", str(tiny_model_path), "--output", str(output), "--passes", "hide-constants"],
        capture_output=True, text=True, cwd=REPO_ROOT, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert output.exists()
