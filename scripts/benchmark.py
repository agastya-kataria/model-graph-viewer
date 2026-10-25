#!/usr/bin/env python3
"""Measure real IR graphs and an explicitly synthetic 5,100-node stress graph."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shlex
import statistics
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from model_graph_viewer import (
    Edge,
    Graph,
    HideConstantConverts,
    HideConstants,
    HighlightNodes,
    Node,
    ReferenceHighFanout,
    TensorMetadata,
    apply_passes,
    load_model,
    render_svg,
)
from model_graph_viewer.metadata import snapshot_graph
from model_graph_viewer.render import to_dot

ROOT = Path(__file__).resolve().parents[1]
SVG = "{http://www.w3.org/2000/svg}"
XLINK = "{http://www.w3.org/1999/xlink}"
ENDPOINT = re.compile(r"(-?\d+):out (\d+) → (-?\d+):in (\d+)")
REFERENCE = re.compile(r"(?:a_)?ref-(-?\d+)-(\d+)-(-?\d+)-(\d+)")
SOURCE = re.compile(r"(?:a_)?source-(-?\d+)-(\d+)")


def endpoint(edge: dict) -> tuple[int, int, int, int]:
    return tuple(edge[key] for key in ("from_node", "from_port", "to_node", "to_port"))


def verify_svg(graph: Graph, svg: Path, original: dict, *, label_detail: str) -> dict[str, object]:
    """Recover connections from the SVG, independently of renderer accounting.

    Graphviz's ordinary edge titles omit the exact ports. The visible edge's
    tooltip retains both ports; a reference's DOM ID encodes the same endpoints.
    Every endpoint tuple must occur exactly once across these representations.
    """
    root = ET.parse(svg).getroot()
    source_uses = {}
    for edge in graph.edges:
        source = (edge.from_node, edge.from_port)
        source_uses[source] = source_uses.get(source, 0) + 1
    node_groups = root.findall(f".//{SVG}g[@class='node']")
    edge_groups = root.findall(f".//{SVG}g[@class='edge']")
    node_ids = [group.get("id") for group in node_groups]
    if len(node_ids) != len(graph.nodes) or set(node_ids) != {
        f"node-{node_id}" for node_id in graph.nodes
    }:
        raise RuntimeError(f"SVG operation identities differ from the graph: {svg}")

    drawn = []
    for group in edge_groups:
        titles = [element.get(f"{XLINK}title", "") for element in group.iter()]
        matches = [match for title in titles if (match := ENDPOINT.fullmatch(title))]
        if len(matches) != 1:
            raise RuntimeError(f"SVG edge lacks one exact endpoint description: {svg}")
        drawn.append(tuple(map(int, matches[0].groups())))

    references = []
    sources = []
    for group in node_groups:
        owner = int(group.get("id").removeprefix("node-"))
        for element in group.iter():
            identifier = element.get("id", "")
            if match := REFERENCE.fullmatch(identifier):
                connection = tuple(map(int, match.groups()))
                if connection[2] != owner:
                    raise RuntimeError(f"Reference is attached to the wrong consumer: {svg}")
                source, port, _, destination_port = connection
                links = list(element.iter(f"{SVG}a"))
                if len(links) != 1 or (
                    links[0].get(f"{XLINK}title")
                    != f"{source}:out {port} → {owner}:in {destination_port}"
                    or links[0].get(f"{XLINK}href") != f"#node-{source}"
                    or f"in {destination_port} ← @{source}:{port}"
                    not in " ".join(element.itertext())
                ):
                    raise RuntimeError(f"Reference text/link differs from its endpoints: {svg}")
                references.append(connection)
            if match := SOURCE.fullmatch(identifier):
                source = tuple(map(int, match.groups()))
                if source[0] != owner:
                    raise RuntimeError(f"Reference marker is attached to the wrong source: {svg}")
                uses = source_uses[source]
                if f"@{source[0]}:{source[1]} ({uses} uses)" not in " ".join(element.itertext()):
                    raise RuntimeError(f"Reference source use count is incorrect: {svg}")
                sources.append(source)

    expected = {
        (edge.from_node, edge.from_port, edge.to_node, edge.to_port): (
            "reference"
            if edge.from_port in graph.nodes[edge.from_node].visual.reference_outputs
            else "edge"
        )
        for edge in graph.edges
    }
    recovered = {connection: "edge" for connection in drawn}
    recovered.update({connection: "reference" for connection in references})
    if len(drawn) + len(references) != len(recovered) or recovered != expected:
        raise RuntimeError(f"SVG connectivity differs from the graph: {svg}")
    expected_sources = {(s, p) for s, p, _, _ in references}
    if len(sources) != len(set(sources)) or set(sources) != expected_sources:
        raise RuntimeError(f"SVG source markers do not match consumer references: {svg}")

    metadata_path = svg.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1:
        raise RuntimeError(f"Unsupported metadata schema: {metadata_path}")
    if any(metadata[key] != original[key] for key in ("nodes", "edges")):
        raise RuntimeError(f"Original graph metadata changed: {metadata_path}")
    current = snapshot_graph(graph)
    if metadata["view"]["nodes"] != current["nodes"]:
        raise RuntimeError(f"View operation metadata differs from the graph: {metadata_path}")
    view_edges = metadata["view"]["edges"]
    if (
        len(view_edges) != len(expected)
        or {endpoint(edge): edge.get("representation") for edge in view_edges} != expected
    ):
        raise RuntimeError(f"View metadata connectivity differs from the graph: {metadata_path}")
    if [
        {key: value for key, value in edge.items() if key != "representation"}
        for edge in view_edges
    ] != current["edges"]:
        raise RuntimeError(f"View tensor metadata differs from the graph: {metadata_path}")
    rendering = metadata["rendering"]
    if (
        rendering["rendered_edges"] != len(drawn)
        or rendering["referenced_edges"] != len(references)
        or rendering["label_detail"] != label_detail
    ):
        raise RuntimeError(f"Rendering counters differ from the SVG: {metadata_path}")

    _, _, width, height = map(float, root.get("viewBox").split())
    widths = []
    for group in node_groups:
        xs = [
            float(point.split(",")[0])
            for polygon in group.findall(f".//{SVG}polygon")
            for point in polygon.get("points", "").split()
        ]
        if xs:
            widths.append(max(xs) - min(xs))
    return {
        "nodes": len(node_groups),
        "logical_edges": len(expected),
        "rendered_edges": len(drawn),
        "referenced_edges": len(references),
        "reference_sources": len({node_id for node_id, _ in sources}),
        "reference_outputs": len(sources),
        "canvas_width_points": width,
        "canvas_height_points": height,
        "node_width_points": {
            "minimum": min(widths) if widths else None,
            "median": statistics.median(widths) if widths else None,
            "maximum": max(widths) if widths else None,
        },
    }


def command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return (result.stdout + result.stderr).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment() -> dict:
    versions = {}
    for distribution in ("model-graph-viewer", "openvino", "numpy", "graphviz"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not installed as a distribution"
    hardware = {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
    }
    if sys.platform == "darwin":
        hardware["model"] = command_output(["sysctl", "-n", "hw.model"])
        hardware["cpu_brand"] = command_output(["sysctl", "-n", "machdep.cpu.brand_string"])
        hardware["memory_bytes"] = command_output(["sysctl", "-n", "hw.memsize"])
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "hardware": hardware,
        "dependencies": versions,
        "graphviz_executable": command_output(["dot", "-V"]),
    }


def synthetic_graph(width: int = 50, layers: int = 100) -> Graph:
    """Independent streams, joined by one shared constant at their first layer.

    The initial graph has 5,101 nodes and 5,100 edges. Constant hiding leaves
    5,100 nodes and 5,050 edges, so SVG layout still handles more than 5,000 nodes.
    This is a scalability fixture, not an OpenVINO model or a neural-network claim.
    """
    tensor = TensorMetadata((1, 64), "f32")
    nodes = [Node(0, "Constant", "shared_offset", output_ports={0: tensor})]
    edges = []
    next_id = 1
    for column in range(width):
        previous = next_id
        nodes.append(Node(next_id, "Parameter", f"input_{column}", output_ports={0: tensor}))
        next_id += 1
        for layer in range(layers):
            inputs = {0: tensor, 1: tensor} if layer == 0 else {0: tensor}
            nodes.append(
                Node(
                    next_id,
                    "Add" if layer == 0 else "Relu",
                    f"stream_{column}/layer_{layer}",
                    input_ports=inputs,
                    output_ports={0: tensor},
                )
            )
            edges.append(Edge(previous, 0, next_id, 0, tensor))
            if layer == 0:
                edges.append(Edge(0, 0, next_id, 1, tensor))
            previous = next_id
            next_id += 1
        nodes.append(Node(next_id, "Result", f"output_{column}", input_ports={0: tensor}))
        edges.append(Edge(previous, 0, next_id, 0, tensor))
        next_id += 1
    return Graph(nodes, edges)


def measure(
    name: str,
    path: Path | None,
    variant: str,
    svg: Path,
    render_timeout: float,
    fanout_threshold: int = 3,
) -> dict:
    start = perf_counter()
    graph = synthetic_graph() if path is None else load_model(path)
    loaded = perf_counter()
    before_nodes, before_edges = len(graph.nodes), len(graph.edges)
    original = snapshot_graph(graph)
    transformations = []
    pass_names = []
    if variant in ("compact", "improved"):
        transformations = [HideConstantConverts(), HideConstants(), HighlightNodes(["projection"])]
        pass_names = ["hide-constant-converts", "hide-constants", "highlight:projection"]
    if variant == "improved":
        transformations.append(ReferenceHighFanout(fanout_threshold))
        pass_names.append("reference-high-fanout")
    apply_passes(graph, transformations)
    transformed = perf_counter()
    label_detail = "full" if variant == "baseline" else "compact"
    render_svg(graph, svg, timeout=render_timeout, label_detail=label_detail, _original=original)
    rendered = perf_counter()
    # Exact graph/SVG/JSON comparisons and geometry measurements are not timed.
    verified = verify_svg(graph, svg, original, label_detail=label_detail)
    metadata_path = svg.with_suffix(".json")
    return {
        "model": name,
        "kind": "synthetic detached graph" if path is None else "real OpenVINO IR",
        "variant": variant,
        "passes": pass_names,
        "label_detail": label_detail,
        "fanout_threshold": fanout_threshold if variant == "improved" else None,
        "before_nodes": before_nodes,
        "before_edges": before_edges,
        "after_nodes": len(graph.nodes),
        "after_edges": len(graph.edges),
        "timings": {
            "load_construct_seconds": loaded - start,
            "transform_seconds": transformed - loaded,
            "render_seconds": rendered - transformed,
        },
        "svg": str(svg.relative_to(ROOT)) if svg.is_relative_to(ROOT) else str(svg),
        "svg_bytes": svg.stat().st_size,
        "metadata": str(metadata_path.relative_to(ROOT))
        if metadata_path.is_relative_to(ROOT)
        else str(metadata_path),
        "metadata_bytes": metadata_path.stat().st_size,
        "svg_verified": verified,
        "layout_graph_attributes": dict(to_dot(graph, label_detail=label_detail).graph_attr),
        "render_timeout_seconds": render_timeout,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", choices=("small", "tinyllama", "all"), default="small")
    parser.add_argument("--include-synthetic", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--render-timeout", type=float, default=300)
    parser.add_argument("--fanout-threshold", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/results.json")
    parser.add_argument("--svg-directory", type=Path, default=ROOT / "examples/svg")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if not math.isfinite(args.render_timeout) or args.render_timeout <= 0:
        parser.error("--render-timeout must be positive and finite")
    if args.fanout_threshold < 1:
        parser.error("--fanout-threshold must be at least 1")
    models = []
    if args.models in ("small", "all"):
        models.extend((name, ROOT / f"examples/models/{name}.xml") for name in ("mlp", "branching"))
    if args.models in ("tinyllama", "all"):
        models.append(("tinyllama", ROOT / "models/tinyllama/openvino_model.xml"))
    for _, path in models:
        if not path.is_file() or not path.with_suffix(".bin").is_file():
            parser.error(
                f"Missing model {path}; run build_examples.py or download_tinyllama.py first"
            )
    if args.include_synthetic:
        models.append(("synthetic_5101", None))
    args.svg_directory.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 2,
        "started_utc": datetime.now(UTC).isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "working_directory": str(Path.cwd()),
        "environment": environment(),
        "methodology": (
            "Single process, independent load/construction for each variant and repetition, "
            "no warm-up; first load includes lazy adapter imports and later runs may benefit "
            "from OS caches. Wall-clock perf_counter. Transform timing includes the original "
            "metadata snapshot, pass bookkeeping, and validation. Render timing includes DOT "
            "generation, Graphviz layout, SVG and JSON writing. Exact layout attributes are "
            "recorded per run. Baseline: no passes, full labels. Compact: Constant/Convert "
            "hiding, compact labels. Improved: compact plus fan-out references. SVG identities, "
            "exact connectivity, reference source markers, JSON snapshots, and tensor metadata "
            "are checked independently after timing. Geometry measurements also occur outside "
            "timings. No compilation or inference. Synthetic renders the improved graph only."
        ),
        "inputs": {
            name: {
                "xml_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "weights_bytes": path.with_suffix(".bin").stat().st_size,
            }
            for name, path in models
            if path is not None
        },
        "runs": [],
    }

    def checkpoint() -> None:
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.output)

    checkpoint()
    try:
        for name, path in models:
            variants = ("improved",) if path is None else ("baseline", "compact", "improved")
            for variant in variants:
                for repetition in range(1, args.repeats + 1):
                    suffix = {"baseline": "before", "compact": "compact", "improved": "after"}[
                        variant
                    ]
                    svg = args.svg_directory / f"{name}.{suffix}.svg"
                    print(f"Measuring {name} / {variant} / repeat {repetition}", flush=True)
                    run = measure(
                        name, path, variant, svg, args.render_timeout, args.fanout_threshold
                    )
                    run["repetition"] = repetition
                    report["runs"].append(run)
                    checkpoint()
                    print(json.dumps(run), flush=True)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        checkpoint()
        raise
    report["completed_utc"] = datetime.now(UTC).isoformat()
    checkpoint()
    print(f"Recorded {len(report['runs'])} measured runs in {args.output}")


if __name__ == "__main__":
    main()
