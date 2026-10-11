"""Public orchestration API shared by scripts and the command-line interface."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol

from .graph import Graph
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
) -> VisualizationResult:
    """Load, transform, and render, measuring the three stages separately."""
    start = perf_counter()
    graph = load_model(model, loader=loader)
    loaded = perf_counter()
    before_nodes, before_edges = len(graph.nodes), len(graph.edges)
    graph = apply_passes(graph, passes)
    transformed = perf_counter()
    path = render_svg(graph, output, timeout=render_timeout)
    rendered = perf_counter()
    return VisualizationResult(
        output_path=str(path),
        before_nodes=before_nodes,
        before_edges=before_edges,
        after_nodes=len(graph.nodes),
        after_edges=len(graph.edges),
        timings={
            "load_construct_seconds": loaded - start,
            "transform_seconds": transformed - loaded,
            "render_seconds": rendered - transformed,
        },
    )
