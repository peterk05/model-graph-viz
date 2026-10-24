# model-graph-viz (Intel OpenVINO/NNCF take-home)

## Source of truth
- The assignment brief is docs/brief.pdf. Re-read the relevant section before each task.
  Exact formats (labels, CLI, pass behaviour) come from the brief, not from memory.
- My own Graph class is in practice/capstone.py. Build on it; don't replace it.

## How I work
- I must be able to explain every line in an interview. Write small, plain Python, no clever tricks.
- I'm still learning Python: explain new syntax in one plain sentence.
- ONE file per task. Don't touch other files unless I ask.
- Don't commit or push; I do git myself.
- Use type hints.
- Only dependencies: openvino, graphviz, pytest.

## Design rules
- loader.py is the only file that imports openvino.
- Edges are (src, src_port, dst, dst_port). Parallel edges must survive.
- Shapes can be dynamic: keep "?" dims, don't crash on them.
- Passes never modify the input graph; they work on a copy.

## Environment
- Python: /home/peterkernohan/openvino_env/bin/python (openvino_env, Python 3.14)
- Test model: models/tinyllama/openvino_model.xml (fp16 export)
- For loader, render, pass or CLI work: run against the test model before saying it's done.

## Docs
- Look up OpenVINO and graphviz APIs with Context7. Name the API you used.
- Never invent URLs.
