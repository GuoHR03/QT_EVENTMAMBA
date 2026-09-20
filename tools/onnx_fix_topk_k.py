"""Normalize scalar TopK K inputs emitted by older PyTorch exporters."""

import argparse

import numpy as np
import onnx
from onnx import numpy_helper


TOPK_PROFILES = {
    "legacy-fps": {
        "group0": (512, 1024, 24),
        "group1": (256, 512, 24),
        "group2": (128, 256, 24),
    },
    "randla-sample": {
        "group0": (512, 24, 24),
        "group1": (256, 24, 24),
        "group2": (128, 24, 24),
    },
}


def normalize_topk_inputs(source, output, profile="legacy-fps"):
    model = onnx.load(source)
    group_sizes = TOPK_PROFILES[profile]
    normalized = 0
    for node in model.graph.node:
        if node.op_type != "TopK":
            continue
        group = node.name.split("/")[1]
        suffix = node.name.rsplit("/TopK", 1)[1]
        position = {"": 0, "_1": 1, "_2": 2}[suffix]
        constant_name = f"{group}_topk_{position}_k"
        model.graph.initializer.append(numpy_helper.from_array(
            np.asarray([group_sizes[group][position]], dtype=np.int64),
            constant_name,
        ))
        node.input[1] = constant_name
        normalized += 1

    onnx.checker.check_model(model)
    onnx.save(model, output)
    return normalized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument(
        "--profile",
        choices=tuple(TOPK_PROFILES),
        default="legacy-fps",
    )
    args = parser.parse_args()

    normalized = normalize_topk_inputs(args.input, args.output, args.profile)
    print(f"normalized {normalized} TopK nodes: {args.output}")


if __name__ == "__main__":
    main()
