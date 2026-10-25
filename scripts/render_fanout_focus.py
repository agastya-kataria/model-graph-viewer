#!/usr/bin/env python3
"""Show fan-out references on a small, real TinyLlama neighborhood."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

from benchmark import SVG, verify_svg

from model_graph_viewer import (
    Graph,
    HideConstantConverts,
    HideConstants,
    HighlightNodes,
    ReferenceHighFanout,
    apply_passes,
    load_model,
    render_svg,
)
from model_graph_viewer.metadata import snapshot_graph

ROOT = Path(__file__).resolve().parents[1]


def focused_graph(graph: Graph, threshold: int = 3) -> tuple[Graph, int, int]:
    """Choose the smallest qualifying fan-out, retaining every direct consumer.

    Ties follow the loader's topological order. This keeps the example compact
    while using genuine operations and all outgoing connections of its source.
    Other external connections are deliberately outside this neighborhood.
    """
    candidates = [
        (len(graph.outgoing(node_id)), order, node_id)
        for order, node_id in enumerate(graph.nodes)
        if len(graph.outgoing(node_id)) > threshold
    ]
    if not candidates:
        raise ValueError(f"No source has more than {threshold} outgoing connections")
    _, _, source = min(candidates)
    selected = {source, *(edge.to_node for edge in graph.outgoing(source))}
    nodes = [deepcopy(node) for node in graph.nodes.values() if node.node_id in selected]
    edges = [
        edge for edge in graph.edges if edge.from_node in selected and edge.to_node in selected
    ]
    omitted = sum(
        (edge.from_node in selected) != (edge.to_node in selected) for edge in graph.edges
    )
    for node in nodes:
        node.attributes["focus"] = "Neighborhood only; external connections omitted"
    return Graph(nodes, edges), source, omitted


def add_boundary_caption(svg: Path, note: str) -> None:
    """Place an explicit boundary note above the renderer's unaltered graph."""
    ET.register_namespace("", SVG[1:-1])
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    tree = ET.parse(svg)
    root = tree.getroot()
    x, y, width, height = map(float, root.get("viewBox").split())
    width = max(width, 680)
    root.set("viewBox", f"{x:g} {y:g} {width:g} {height + 36:g}")
    root.set("width", f"{width:g}pt")
    root.set("height", f"{height + 36:g}pt")
    content = ET.Element(f"{SVG}g", {"transform": "translate(0 36)"})
    for child in list(root):
        root.remove(child)
        content.append(child)
    ET.SubElement(
        root,
        f"{SVG}rect",
        {
            "x": str(x),
            "y": str(y),
            "width": str(width),
            "height": str(height + 36),
            "fill": "white",
        },
    )
    caption = ET.SubElement(
        root,
        f"{SVG}text",
        {"x": "8", "y": "16", "font-family": "Helvetica", "font-size": "10"},
    )
    caption.text = note
    root.append(content)
    tree.write(svg, encoding="utf-8", xml_declaration=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models/tinyllama/openvino_model.xml")
    parser.add_argument("--output-directory", type=Path, default=ROOT / "examples/svg")
    parser.add_argument("--fanout-threshold", type=int, default=3)
    args = parser.parse_args()
    if args.fanout_threshold < 1:
        parser.error("--fanout-threshold must be at least 1")
    graph = load_model(args.model)
    apply_passes(graph, [HideConstantConverts(), HideConstants()])
    focus, source, omitted = focused_graph(graph, args.fanout_threshold)
    original = snapshot_graph(focus)
    HighlightNodes([focus.nodes[source].name]).apply(focus)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    outcomes = {}
    for variant in ("before", "after"):
        if variant == "after":
            ReferenceHighFanout(args.fanout_threshold).apply(focus)
        output = args.output_directory / f"tinyllama.fanout.focus.{variant}.svg"
        render_svg(focus, output, _original=original)
        add_boundary_caption(
            output,
            f"Source {source} and all direct consumers. "
            f"Neighborhood boundary: {omitted} external connections omitted.",
        )
        outcomes[variant] = verify_svg(focus, output, original, label_detail="compact")
    report = {
        "selection": "Smallest qualifying fan-out; ties follow topological graph order",
        "preparation": ["hide-constant-converts", "hide-constants"],
        "selected_node_id": source,
        "selected_name": focus.nodes[source].name,
        "fanout_threshold": args.fanout_threshold,
        "before": outcomes["before"],
        "after": outcomes["after"],
        "boundary": f"{omitted} external connections intentionally omitted in both views",
        "metadata_scope": "Original and view snapshots describe this prepared neighborhood",
    }
    (args.output_directory / "tinyllama.fanout.focus.summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
