"""Build the two small example OpenVINO models used in examples/.

They are built in code with the OpenVINO Python API (no downloads needed) and saved
as normal OpenVINO IR (.xml + .bin). Each one is designed to exercise cases from the
brief:

small_cnn
    Convolutions, a Split with several outputs, two branches joined by Concat,
    a Constant shared by two consumers, and an Add that takes the same tensor twice
    (two edges between the same pair of nodes).

mini_attention
    A single-head attention block like one TinyLlama layer, with fp16 weights that are
    decompressed by Convert ops (like TinyLlama's fp16 export), dynamic batch and
    sequence dimensions, two model outputs, and two nodes with the same name.

Usage:
    python scripts/build_example_models.py            # writes examples/models/...
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import openvino as ov
import openvino.opset13 as ops

OUT_DIR = Path(__file__).resolve().parent.parent / "examples" / "models"


def build_small_cnn() -> ov.Model:
    rng = np.random.default_rng(0)
    image = ops.parameter([1, 3, 32, 32], ov.Type.f32, name="image")

    conv1 = ops.convolution(image, ops.constant(rng.standard_normal((16, 3, 3, 3)).astype(np.float32)),
                            strides=[1, 1], pads_begin=[1, 1], pads_end=[1, 1], dilations=[1, 1])
    conv1.set_friendly_name("conv1")
    bias = ops.constant(rng.standard_normal((1, 16, 1, 1)).astype(np.float32))
    bias.set_friendly_name("shared_bias")
    biased = ops.add(conv1, bias)
    relu1 = ops.relu(biased)
    pool = ops.max_pool(relu1, strides=[2, 2], dilations=[1, 1], pads_begin=[0, 0], pads_end=[0, 0],
                        kernel_shape=[2, 2])
    pool.set_friendly_name("pool")

    # Split into two halves along channels: one op, two output ports.
    halves = ops.split(pool.output(0), ops.constant(np.int64(1)), 2)
    halves.set_friendly_name("split_channels")

    left = ops.convolution(halves.output(0), ops.constant(rng.standard_normal((8, 8, 3, 3)).astype(np.float32)),
                           strides=[1, 1], pads_begin=[1, 1], pads_end=[1, 1], dilations=[1, 1])
    left.set_friendly_name("branch_a")
    right = ops.convolution(halves.output(1), ops.constant(rng.standard_normal((8, 8, 1, 1)).astype(np.float32)),
                            strides=[1, 1], pads_begin=[0, 0], pads_end=[0, 0], dilations=[1, 1])
    right.set_friendly_name("branch_b")
    # Same bias constant reused: a constant with several consumers.
    left_biased = ops.add(left, ops.slice(bias, np.array([0, 0, 0, 0]), np.array([1, 8, 1, 1]), np.array([1, 1, 1, 1])))
    doubled = ops.add(right, right)   # the same tensor on both inputs -> two parallel edges
    doubled.set_friendly_name("double_b")

    merged = ops.concat([left_biased, doubled], axis=1)
    merged.set_friendly_name("merge")
    pooled = ops.reduce_mean(merged, ops.constant(np.array([2, 3], dtype=np.int64)), keep_dims=False)
    pooled.set_friendly_name("global_pool")
    logits = ops.matmul(pooled, ops.constant(rng.standard_normal((16, 10)).astype(np.float32)), False, False)
    logits.set_friendly_name("classifier")
    probs = ops.softmax(logits, axis=1)
    probs.set_friendly_name("probs")
    return ov.Model([ops.result(probs)], [image], "small_cnn")


def build_mini_attention() -> ov.Model:
    rng = np.random.default_rng(1)
    hidden = 64
    x = ops.parameter([-1, -1, hidden], ov.Type.f32, name="hidden_states")   # [batch?, seq?, 64]

    def fp16_weight(shape: tuple[int, ...], name: str) -> ov.Node:
        """fp16 Constant + Convert to fp32: the weight-compression pattern of TinyLlama."""
        const = ops.constant(rng.standard_normal(shape).astype(np.float16))
        const.set_friendly_name(f"{name}.weight")
        convert = ops.convert(const, ov.Type.f32)
        convert.set_friendly_name(f"{name}.weight/decompress")
        return convert

    # RMSNorm, written with the same op types TinyLlama uses.
    squared = ops.power(x, ops.constant(np.float32(2.0)))
    mean = ops.reduce_mean(squared, ops.constant(np.int64(-1)), keep_dims=True)
    root = ops.sqrt(ops.add(mean, ops.constant(np.float32(1e-5))))
    normed = ops.multiply(ops.divide(x, root), fp16_weight((hidden,), "norm"))
    normed.set_friendly_name("input_layernorm")

    q = ops.matmul(normed, fp16_weight((hidden, hidden), "q_proj"), False, True)
    k = ops.matmul(normed, fp16_weight((hidden, hidden), "k_proj"), False, True)
    v = ops.matmul(normed, fp16_weight((hidden, hidden), "v_proj"), False, True)
    q.set_friendly_name("q_proj")
    k.set_friendly_name("k_proj")
    v.set_friendly_name("v_proj")

    attention = ops.scaled_dot_product_attention(q, k, v, causal=True)
    attention.set_friendly_name("attention")
    o = ops.matmul(attention, fp16_weight((hidden, hidden), "o_proj"), False, True)
    o.set_friendly_name("o_proj")
    residual = ops.add(x, o)
    residual.set_friendly_name("residual")   # duplicate name on purpose (see below)

    up = ops.matmul(residual, fp16_weight((4 * hidden, hidden), "up_proj"), False, True)
    gate = ops.swish(ops.matmul(residual, fp16_weight((4 * hidden, hidden), "gate_proj"), False, True))
    down = ops.matmul(ops.multiply(gate, up), fp16_weight((hidden, 4 * hidden), "down_proj"), False, True)
    out = ops.add(residual, down)
    out.set_friendly_name("residual")        # second node called "residual"

    return ov.Model([ops.result(out), ops.result(attention)], [x], "mini_attention")


def main() -> None:
    for name, builder in [("small_cnn", build_small_cnn), ("mini_attention", build_mini_attention)]:
        target = OUT_DIR / name / f"{name}.xml"
        target.parent.mkdir(parents=True, exist_ok=True)
        ov.save_model(builder(), target, compress_to_fp16=False)
        print(f"wrote {target.relative_to(OUT_DIR.parent.parent)}")


if __name__ == "__main__":
    main()
