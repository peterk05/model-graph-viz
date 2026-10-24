#!/usr/bin/env bash
# One-shot: install deps, tidy old placeholder files, run the tests, generate every
# example SVG + the TinyLlama benchmark, then commit and push.
#
#   source ~/openvino_env/bin/activate
#   bash scripts/finish_submission.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/6 dependencies"
python -m pip install --quiet -r requirements.txt

echo "== 2/6 remove old empty placeholder files (replaced by packages)"
for f in mgv/loader.py mgv/render.py mgv/passes.py run.py tests/test_loader.py; do
  if [ -f "$f" ]; then git rm -q --cached "$f" 2>/dev/null || true; rm -f "$f"; echo "removed $f"; fi
done
if [ -f explore.py ]; then mkdir -p practice; git rm -q --cached explore.py 2>/dev/null || true; mv explore.py practice/; echo "moved explore.py to practice/"; fi
find . -name "__pycache__" -type d -prune -exec rm -rf {} +

echo "== 3/6 TinyLlama model"
bash scripts/export_tinyllama.sh

echo "== 4/6 tests"
python -m pytest -q

echo "== 5/6 example models, SVGs and benchmark (a few minutes: Graphviz is slow on the full TinyLlama graph)"
python scripts/build_example_models.py
python scripts/make_examples.py

echo "== 6/6 commit and push"
git add -A
git commit -m "Model graph visualization: library, CLI, passes, tests, examples, benchmark" || echo "nothing new to commit"
git push
echo "done: https://github.com/peterk05/model-graph-viz"
