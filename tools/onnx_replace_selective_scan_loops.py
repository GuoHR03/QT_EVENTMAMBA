"""Replace exported selective-scan Loop nodes with the Windows CUDA custom op."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import onnx
from onnx import helper


def _captured_tensor(body, suffix):
    produced = {output for node in body.node for output in node.output}
    body_inputs = {value.name for value in body.input}
    candidates = []
    for node in body.node:
        for value in node.input:
            if value and value not in produced and value not in body_inputs:
                if suffix in value and value not in candidates:
                    candidates.append(value)
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one captured {suffix} tensor, got {candidates}")
    return candidates[0]


def replace_loops(source, output):
    model = onnx.load(source)
    consumers = defaultdict(list)
    for node in model.graph.node:
        for value in node.input:
            consumers[value].append(node)

    replacements = {}
    skipped = set()
    details = []
    for loop in model.graph.node:
        if loop.op_type != "Loop":
            continue
        body = next(attribute.g for attribute in loop.attribute if attribute.name == "body")
        sequence_output = loop.output[1]
        sequence_consumers = consumers[sequence_output]
        if len(sequence_consumers) != 1 or sequence_consumers[0].op_type != "ConcatFromSequence":
            raise RuntimeError(f"Unexpected Loop consumer for {loop.name}")
        concat = sequence_consumers[0]
        delta_a = _captured_tensor(body, "/Exp")
        delta_b_u = _captured_tensor(body, "/Einsum_1")
        c_term = _captured_tensor(body, "/Cast_3")
        replacement = helper.make_node(
            "SelectiveScanCore",
            [delta_a, delta_b_u, c_term],
            list(concat.output),
            domain="com.eventmamba",
            name=f"{loop.name}_cuda",
        )
        replacements[loop.name] = replacement
        skipped.add(concat.name)
        details.append(
            {
                "loop": loop.name,
                "inputs": [delta_a, delta_b_u, c_term],
                "output": concat.output[0],
            }
        )

    if len(replacements) != 6:
        raise RuntimeError(f"Expected 6 selective scan loops, found {len(replacements)}")
    rewritten = []
    for node in model.graph.node:
        if node.name in replacements:
            rewritten.append(replacements[node.name])
        elif node.name not in skipped:
            rewritten.append(node)
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    if not any(item.domain == "com.eventmamba" for item in model.opset_import):
        model.opset_import.append(helper.make_opsetid("com.eventmamba", 1))
    onnx.checker.check_model(model)
    onnx.save(model, output)
    return details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    details = replace_loops(args.input, args.output)
    print(
        json.dumps(
            {
                "status": "rewritten",
                "input": str(Path(args.input).resolve()),
                "output": str(Path(args.output).resolve()),
                "replacements": details,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
