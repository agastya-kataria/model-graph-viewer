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
import shlex
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
    TensorMetadata,
    apply_passes,
    load_model,
    render_svg,
)
from model_graph_viewer.render import to_dot

ROOT = Path(__file__).resolve().parents[1]


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


def measure(name: str, path: Path | None, variant: str, svg: Path, render_timeout: float) -> dict:
    start = perf_counter()
    graph = synthetic_graph() if path is None else load_model(path)
    loaded = perf_counter()
    before_nodes, before_edges = len(graph.nodes), len(graph.edges)
    transformations = []
    if variant == "improved":
        transformations = [HideConstantConverts(), HideConstants(), HighlightNodes(["projection"])]
    apply_passes(graph, transformations)
    transformed = perf_counter()
    render_svg(graph, svg, timeout=render_timeout)
    rendered = perf_counter()
    # Count rendered operation/connection groups, outside the stage timings.
    groups = ET.parse(svg).getroot().findall(".//{http://www.w3.org/2000/svg}g")
    rendered_counts = {
        "nodes": sum(group.get("class") == "node" for group in groups),
        "edges": sum(group.get("class") == "edge" for group in groups),
    }
    if rendered_counts != {"nodes": len(graph.nodes), "edges": len(graph.edges)}:
        raise RuntimeError(f"SVG completeness check failed for {svg}: {rendered_counts}")
    return {
        "model": name,
        "kind": "synthetic detached graph" if path is None else "real OpenVINO IR",
        "variant": variant,
        "passes": []
        if variant == "baseline"
        else ["hide-constant-converts", "hide-constants", "highlight:projection"],
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
        "svg_verified_counts": rendered_counts,
        "layout_graph_attributes": dict(to_dot(graph).graph_attr),
        "render_timeout_seconds": render_timeout,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", choices=("small", "tinyllama", "all"), default="small")
    parser.add_argument("--include-synthetic", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--render-timeout", type=float, default=300)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/results.json")
    parser.add_argument("--svg-directory", type=Path, default=ROOT / "examples/svg")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if not math.isfinite(args.render_timeout) or args.render_timeout <= 0:
        parser.error("--render-timeout must be positive and finite")
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
        "schema_version": 1,
        "started_utc": datetime.now(UTC).isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "working_directory": str(Path.cwd()),
        "environment": environment(),
        "methodology": (
            "Single process, independent load/construction for each variant and repetition, "
            "no warm-up; first load includes lazy adapter imports and later runs may benefit "
            "from OS caches. Wall-clock perf_counter. Transform timing includes bookkeeping "
            "and validation. Render timing includes DOT generation, Graphviz layout, and SVG "
            "writing. More than 1,000 nodes use Graphviz nslimit=2, nslimit1=2, mclimit=0.1, "
            "splines=line; smaller graphs use dot defaults. Exact layout attributes are "
            "recorded per run. Rendering has a 300-second default timeout (configurable). "
            "SVG node/edge completeness is checked after timing. No compilation or inference. "
            "Synthetic renders the improved graph only."
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
            variants = ("improved",) if path is None else ("baseline", "improved")
            for variant in variants:
                for repetition in range(1, args.repeats + 1):
                    suffix = "before" if variant == "baseline" else "after"
                    svg = args.svg_directory / f"{name}.{suffix}.svg"
                    print(f"Measuring {name} / {variant} / repeat {repetition}", flush=True)
                    run = measure(name, path, variant, svg, args.render_timeout)
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
