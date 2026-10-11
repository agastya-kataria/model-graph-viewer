"""CLI configuration and error reporting; pipeline behavior lives in the API."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

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
    if args.stats and args.stats.resolve() in {args.model.resolve(), args.output.resolve()}:
        parser.error("--stats must be different from the model and SVG paths")
    try:
        passes = [
            create_pass(name, **({"names": args.highlight} if name == "highlight" else {}))
            for name in names
        ]
        result = visualize(args.model, args.output, passes, render_timeout=args.render_timeout)
        if args.stats:
            args.stats.parent.mkdir(parents=True, exist_ok=True)
            args.stats.write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")
    except (ValueError, RenderError, OSError, ImportError) as exc:
        print(f"model-graph-viewer: error: {exc}", file=sys.stderr)
        return 1
    print(
        f"Wrote {result.output_path}: "
        f"{result.before_nodes} → {result.after_nodes} nodes, "
        f"{result.before_edges} → {result.after_edges} edges"
    )
    return 0
