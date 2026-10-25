"""Public orchestration API shared by scripts and the command-line interface."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol

from ._paths import validate_output_paths
from .graph import Graph
from .metadata import edge_counts, snapshot_graph
from .passes import TransformationPass, apply_passes
from .render import render_svg


class ModelLoader(Protocol):
    """An adapter produces a detached graph; no renderer or pass changes are needed."""

    def load(self, path: str | Path) -> Graph: ...


@dataclass(frozen=True)
class VisualizationResult:
    output_path: str
    before_nodes: int
    before_edges: int
    after_nodes: int
    after_edges: int
    timings: dict[str, float]
    metadata_path: str
    rendered_edges: int
    referenced_edges: int


def load_model(path: str | Path, *, loader: ModelLoader | None = None) -> Graph:
    """Load OpenVINO IR, or use an explicitly supplied adapter for another format."""
    if loader is None:
        from .openvino_loader import OpenVINOLoader

        loader = OpenVINOLoader()
    graph = loader.load(path)
    graph.validate()
    return graph


def visualize(
    model: str | Path,
    output: str | Path,
    passes: Iterable[TransformationPass] = (),
    *,
    loader: ModelLoader | None = None,
    render_timeout: float = 300,
    label_detail: str = "compact",
    metadata: str | Path | None = None,
) -> VisualizationResult:
    """Load, transform, and export an SVG plus original/view metadata.

    Compact labels omit names and verbose attributes. The JSON sidecar retains
    the graph before transformations as well as the resulting visualization.
    ``after_edges`` counts logical connections; references are counted separately.
    """
    if label_detail not in {"compact", "full"}:
        raise ValueError("label_detail must be 'compact' or 'full'")
    model_path, output_path, _, metadata_path = validate_output_paths(
        model, output, metadata=metadata
    )
    start = perf_counter()
    graph = load_model(model_path, loader=loader)
    loaded = perf_counter()
    before_nodes, before_edges = len(graph.nodes), len(graph.edges)
    original = snapshot_graph(graph)
    graph = apply_passes(graph, passes)
    drawn, referenced = edge_counts(graph)
    transformed = perf_counter()
    path = render_svg(
        graph,
        output_path,
        timeout=render_timeout,
        label_detail=label_detail,
        metadata=metadata_path,
        _original=original,
    )
    rendered = perf_counter()
    return VisualizationResult(
        output_path=str(path),
        before_nodes=before_nodes,
        before_edges=before_edges,
        after_nodes=len(graph.nodes),
        after_edges=len(graph.edges),
        metadata_path=str(metadata_path),
        rendered_edges=drawn,
        referenced_edges=referenced,
        timings={
            "load_construct_seconds": loaded - start,
            "transform_seconds": transformed - loaded,
            "render_seconds": rendered - transformed,
        },
    )
