#!/usr/bin/env python3
"""Create the two small OpenVINO IR examples with a fixed random seed."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import openvino as ov
from openvino import opset13 as ops

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261010


def mlp() -> ov.Model:
    """The starter's MatMul/Add graph, with a correctly broadcast row bias."""
    rng = np.random.default_rng(SEED)
    data = ops.parameter([1, 768], np.float32, name="input")
    weights = ops.constant(rng.normal(0, 0.02, (768, 1024)).astype(np.float32), name="weights")
    projection = ops.matmul(data, weights, False, False, name="projection")
    bias = ops.constant(np.zeros((1, 1024), np.float32), name="bias")
    output = ops.add(projection, bias, name="biased_projection")
    result = ops.result(output, name="output")
    return ov.Model([result], [data], "starter_mlp")


def branching() -> ov.Model:
    """Keep distinct ports and repeated names visible in a compact real model."""
    rng = np.random.default_rng(SEED)
    data = ops.parameter([-1, 8], np.float32, name="dynamic_input")
    axis = ops.constant(np.array(1, dtype=np.int64), name="split_axis")
    split = ops.split(data, axis, 2, name="feature_split")
    weights = ops.constant(rng.normal(0, 0.1, (4, 4)).astype(np.float32), name="shared_weights")
    left = ops.matmul(split.output(0), weights, False, False, name="projection")
    right = ops.matmul(split.output(1), weights, False, False, name="projection")
    # Two edges from left:0 to separate ports of one consumer must remain distinct.
    doubled = ops.add(left, left, name="doubled_left")
    combined = ops.add(doubled, right, name="combined")
    result = ops.result(combined, name="output")
    return ov.Model([result], [data], "branching_dynamic_batch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=ROOT / "examples/models")
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    for name, builder in (("mlp", mlp), ("branching", branching)):
        model = builder()
        path = args.output_directory / f"{name}.xml"
        # serialize preserves graph structure and float32 values; no optimization/inference.
        ov.serialize(model, str(path), str(path.with_suffix(".bin")))
        restored = ov.Core().read_model(str(path))
        nodes = restored.get_ordered_ops()
        print(f"{path}: {len(nodes)} nodes, {sum(len(node.inputs()) for node in nodes)} edges")


if __name__ == "__main__":
    main()
