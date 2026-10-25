"""Portable graph metadata accompanying a visualization, without tensor values."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from pathlib import Path

from .graph import Graph, TensorMetadata


def metadata_path_for(output: str | Path, metadata: str | Path | None = None) -> Path:
    """Use the SVG's stem unless a separate metadata destination was requested."""
    path = Path(metadata) if metadata is not None else Path(output).with_suffix(".json")
    try:
        return path.expanduser()
    except RuntimeError as exc:
        raise ValueError(f"Cannot expand path '{path}': {exc}") from exc


def _portable(value: object) -> object:
    """Copy supported attributes into detached JSON values."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        result = float(value)
        return result if math.isfinite(result) else str(result)
    if isinstance(value, Mapping):
        return {str(key): _portable(item) for key, item in value.items()}
    if isinstance(value, Sequence):
        return [_portable(item) for item in value]
    return str(value)


def _tensor(tensor: TensorMetadata) -> dict:
    return {"shape": _portable(tensor.shape), "dtype": tensor.dtype}


def snapshot_graph(graph: Graph) -> dict:
    """Capture graph identities, attributes, topology, and display annotations.

    The result does not retain mutable graph objects. Constant tensor contents
    are never read; only metadata already present in the detached graph is used.
    """
    graph.validate()
    nodes = {}
    for node_id, node in graph.nodes.items():
        hidden = {
            str(port): {
                "source_id": annotation.source_id,
                "tensor": _tensor(annotation.tensor),
                "shared": annotation.shared,
                "convert_id": annotation.convert_id,
                "stored_dtype": annotation.stored_dtype,
                "source_shared": annotation.source_shared,
            }
            for port, annotation in sorted(node.visual.hidden_inputs.items())
        }
        nodes[str(node_id)] = {
            "node_id": node_id,
            "node_type": node.node_type,
            "name": node.name,
            "attributes": _portable(node.attributes),
            "input_ports": {
                str(port): _tensor(tensor) for port, tensor in node.input_ports.items()
            },
            "output_ports": {
                str(port): _tensor(tensor) for port, tensor in node.output_ports.items()
            },
            "visual": {
                "border_color": node.visual.border_color,
                "constant_shared": node.visual.constant_shared,
                "reference_outputs": sorted(node.visual.reference_outputs),
                "hidden_inputs": hidden,
            },
        }
    edges = [
        {
            "from_node": edge.from_node,
            "from_port": edge.from_port,
            "to_node": edge.to_node,
            "to_port": edge.to_port,
            "tensor": _tensor(edge.tensor),
        }
        for edge in graph.edges
    ]
    return {"nodes": nodes, "edges": edges}


def edge_counts(graph: Graph) -> tuple[int, int]:
    """Return the numbers of drawn and referenced graph connections."""
    referenced = sum(
        edge.from_port in graph.nodes[edge.from_node].visual.reference_outputs
        for edge in graph.edges
    )
    return len(graph.edges) - referenced, referenced


def visualization_metadata(graph: Graph, label_detail: str, original: dict | None = None) -> dict:
    """Describe both the original graph and its current rendered representation."""
    current = snapshot_graph(graph)
    source = current if original is None else original
    view_edges = [
        {
            **record,
            "representation": (
                "reference"
                if edge.from_port in graph.nodes[edge.from_node].visual.reference_outputs
                else "edge"
            ),
        }
        for edge, record in zip(graph.edges, current["edges"], strict=True)
    ]
    drawn, referenced = edge_counts(graph)
    return {
        "schema_version": 1,
        "nodes": source["nodes"],
        "edges": source["edges"],
        "view": {"nodes": current["nodes"], "edges": view_edges},
        "rendering": {
            "label_detail": label_detail,
            "rendered_edges": drawn,
            "referenced_edges": referenced,
        },
    }
