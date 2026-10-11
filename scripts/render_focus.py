#!/usr/bin/env python3
"""Render a reproducible TinyLlama neighborhood around a converted MatMul weight."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from model_graph_viewer import (
    Graph,
    HideConstantConverts,
    HideConstants,
    HighlightNodes,
    apply_passes,
    load_model,
    render_svg,
)

ROOT = Path(__file__).resolve().parents[1]


def focused_graph(graph: Graph) -> tuple[Graph, int]:
    """Pick the first MatMul with a direct Constant -> Convert weight input.

    Copy the operation, all its direct input sources, upstream Constants for
    those Converts, and direct output consumers. Boundary nodes keep their
    original ports but the focus intentionally omits their external edges.
    """
    for node in graph.nodes.values():
        if node.node_type != "MatMul":
            continue
        matched = False
        for edge in graph.incoming(node.node_id):
            producer = graph.nodes[edge.from_node]
            if edge.to_port != 1 or producer.node_type != "Convert":
                continue
            inputs = tuple(graph.incoming(producer.node_id))
            if len(inputs) == 1 and graph.nodes[inputs[0].from_node].node_type == "Constant":
                matched = True
                break
        if not matched:
            continue
        selected = {node.node_id}
        for edge in graph.incoming(node.node_id):
            selected.add(edge.from_node)
            producer = graph.nodes[edge.from_node]
            if producer.node_type == "Convert":
                selected.update(upstream.from_node for upstream in graph.incoming(producer.node_id))
        selected.update(edge.to_node for edge in graph.outgoing(node.node_id))
        nodes = [deepcopy(source) for node_id, source in graph.nodes.items() if node_id in selected]
        for copied in nodes:
            if copied.node_id == node.node_id:
                copied.attributes["focus"] = "Selected MatMul; external edges omitted at boundary"
            else:
                copied.attributes["focus"] = "Neighborhood boundary may omit external edges"
        edges = [
            edge for edge in graph.edges if edge.from_node in selected and edge.to_node in selected
        ]
        return Graph(nodes, edges), node.node_id
    raise ValueError("No MatMul with a direct Constant -> Convert weight branch was found")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models/tinyllama/openvino_model.xml")
    parser.add_argument("--output-directory", type=Path, default=ROOT / "examples/svg")
    args = parser.parse_args()
    graph = load_model(args.model)
    focus, selected = focused_graph(graph)
    name = focus.nodes[selected].name
    args.output_directory.mkdir(parents=True, exist_ok=True)
    render_svg(focus, args.output_directory / "tinyllama.focus.before.svg")
    before = {"nodes": len(focus.nodes), "edges": len(focus.edges)}
    apply_passes(focus, [HideConstantConverts(), HideConstants(), HighlightNodes([name])])
    render_svg(focus, args.output_directory / "tinyllama.focus.after.svg")
    report = {
        "selection": "First MatMul in topological graph order with Constant -> Convert on input 1",
        "selected_node_id": selected,
        "selected_name": name,
        "before": before,
        "after": {"nodes": len(focus.nodes), "edges": len(focus.edges)},
        "boundary": (
            "External connections are intentionally omitted; node port metadata is preserved."
        ),
    }
    (args.output_directory / "tinyllama.focus.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
