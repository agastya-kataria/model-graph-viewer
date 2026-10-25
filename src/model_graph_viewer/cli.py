"""CLI configuration and error reporting; pipeline behavior lives in the API."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from ._paths import atomic_write_text, validate_output_paths
from .api import visualize
from .passes import PASS_REGISTRY, create_pass
from .render import RenderError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Visualize an OpenVINO IR model as a port-aware SVG graph."
    )
    parser.add_argument("--model", required=True, type=Path, help="OpenVINO .xml model path")
    parser.add_argument("--output", required=True, type=Path, help="Destination .svg path")
    parser.add_argument(
        "--passes",
        default="",
        help=f"Comma-separated passes in execution order: {', '.join(PASS_REGISTRY)}",
    )
    parser.add_argument(
        "--highlight",
        action="append",
        default=[],
        metavar="NAME",
        help="Exact node name for the highlight pass; repeat for multiple names",
    )
    parser.add_argument("--stats", type=Path, help="Also write counts and stage timings as JSON")
    parser.add_argument(
        "--metadata", type=Path, help="Metadata JSON path (default: SVG basename with .json)"
    )
    parser.add_argument(
        "--label-detail",
        choices=("compact", "full"),
        default="compact",
        help="Visible label detail (default: compact; full includes names and verbose attributes)",
    )
    parser.add_argument(
        "--fanout-threshold",
        type=int,
        metavar="N",
        help="Reference nodes with more than N outgoing edges (default: 3; requires pass)",
    )
    parser.add_argument(
        "--render-timeout",
        type=float,
        default=300,
        metavar="SECONDS",
        help="Maximum Graphviz running time in seconds (default: 300)",
    )
    args = parser.parse_args(argv)
    names = [name.strip() for name in args.passes.split(",")] if args.passes else []
    if any(not name for name in names):
        parser.error("--passes must not contain empty pass names")
    if args.highlight and "highlight" not in names:
        parser.error("--highlight requires 'highlight' in --passes")
    if "highlight" in names and not args.highlight:
        parser.error("the highlight pass requires at least one --highlight NAME")
    if args.fanout_threshold is not None:
        if args.fanout_threshold < 1:
            parser.error("--fanout-threshold must be a positive integer")
        if "reference-high-fanout" not in names:
            parser.error("--fanout-threshold requires 'reference-high-fanout' in --passes")
    try:
        args.model, args.output, args.stats, args.metadata = validate_output_paths(
            args.model, args.output, args.stats, args.metadata
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    try:
        passes = []
        for name in names:
            options = {}
            if name == "highlight":
                options["names"] = args.highlight
            elif name == "reference-high-fanout" and args.fanout_threshold is not None:
                options["threshold"] = args.fanout_threshold
            passes.append(create_pass(name, **options))
        result = visualize(
            args.model,
            args.output,
            passes,
            render_timeout=args.render_timeout,
            label_detail=args.label_detail,
            metadata=args.metadata,
        )
        if args.stats:
            atomic_write_text(args.stats, json.dumps(asdict(result), indent=2) + "\n")
    except (ValueError, RenderError, OSError, ImportError) as exc:
        print(f"model-graph-viewer: error: {exc}", file=sys.stderr)
        return 1
    print(
        f"Wrote {result.output_path}: "
        f"{result.before_nodes} → {result.after_nodes} nodes, "
        f"{result.before_edges} → {result.after_edges} logical edges "
        f"({result.rendered_edges} drawn, {result.referenced_edges} referenced); "
        f"metadata: {result.metadata_path}"
    )
    return 0
