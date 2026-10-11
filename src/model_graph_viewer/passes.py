"""Ordered transformations of detached visualization graphs."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Protocol

from .graph import ConstantAnnotation, Edge, Graph, TensorMetadata


class TransformationPass(Protocol):
    def apply(self, graph: Graph) -> Graph:
        """Mutate and return the supplied graph."""
        ...


class HighlightNodes:
    """Highlight all exact name matches, including duplicate names."""

    def __init__(self, names: Iterable[str] = ()) -> None:
        self.names = frozenset(names)

    def apply(self, graph: Graph) -> Graph:
        for node in graph.nodes.values():
            if node.name in self.names:
                node.visual.border_color = "red"
        return graph


def _remember_constant_sharing(graph: Graph) -> None:
    for node in graph.nodes.values():
        if node.node_type == "Constant":
            consumers = {edge.to_node for edge in graph.outgoing(node.node_id)}
            node.visual.constant_shared |= len(consumers) > 1


def _edge_tensor(graph: Graph, edge: Edge) -> TensorMetadata:
    """Use connection metadata, filling missing fields from its source port."""
    port = graph.nodes[edge.from_node].output_ports[edge.from_port]
    return TensorMetadata(
        shape=edge.tensor.shape if edge.tensor.shape is not None else port.shape,
        dtype=edge.tensor.dtype if edge.tensor.dtype is not None else port.dtype,
    )


def _remove_if_disconnected(graph: Graph, node_id: int) -> None:
    if not graph.incoming(node_id) and not graph.outgoing(node_id):
        graph.remove_node(node_id)


class HideConstants:
    """Replace visible Constant connections with annotations at original ports."""

    def apply(self, graph: Graph) -> Graph:
        _remember_constant_sharing(graph)
        constants = [node for node in graph.nodes.values() if node.node_type == "Constant"]
        for constant in constants:
            for edge in tuple(graph.outgoing(constant.node_id)):
                annotation = ConstantAnnotation(
                    source_id=constant.node_id,
                    tensor=_edge_tensor(graph, edge),
                    shared=constant.visual.constant_shared,
                    source_shared=constant.visual.constant_shared,
                )
                graph.remove_edge(edge)
                graph.nodes[edge.to_node].visual.hidden_inputs[edge.to_port] = annotation
        # Remove nodes only after all connections are hidden. This also handles
        # synthetic Constants with incoming edges, regardless of node order.
        for constant in constants:
            _remove_if_disconnected(graph, constant.node_id)
        return graph


class HideConstantConverts:
    """Fold only visible, direct Constant -> Convert branches into inputs.

    Converted values have their own identities and sharing counts. The source
    identity and storage dtype remain available without reading tensor values.
    Running this after HideConstants intentionally leaves Converts visible.
    """

    def apply(self, graph: Graph) -> Graph:
        _remember_constant_sharing(graph)
        candidates = [node for node in graph.nodes.values() if node.node_type == "Convert"]
        for convert in candidates:
            if len(convert.input_ports) != 1 or len(convert.output_ports) != 1:
                continue
            incoming = tuple(graph.incoming(convert.node_id))
            if len(incoming) != 1:
                continue
            source_edge = incoming[0]
            source = graph.nodes[source_edge.from_node]
            if source.node_type != "Constant":
                continue
            outgoing = tuple(graph.outgoing(convert.node_id))
            shared = len({edge.to_node for edge in outgoing}) > 1
            stored_dtype = _edge_tensor(graph, source_edge).dtype
            for edge in outgoing:
                annotation = ConstantAnnotation(
                    source_id=source.node_id,
                    tensor=_edge_tensor(graph, edge),
                    shared=shared,
                    convert_id=convert.node_id,
                    stored_dtype=stored_dtype,
                    source_shared=source.visual.constant_shared,
                )
                graph.remove_edge(edge)
                graph.nodes[edge.to_node].visual.hidden_inputs[edge.to_port] = annotation
            graph.remove_edge(source_edge)
            _remove_if_disconnected(graph, convert.node_id)
            _remove_if_disconnected(graph, source.node_id)
        return graph


def apply_passes(graph: Graph, passes: Iterable[TransformationPass]) -> Graph:
    """Apply exactly the supplied order, validating every resulting graph."""
    graph.validate()
    for transformation in passes:
        if transformation.apply(graph) is not graph:
            raise ValueError("Graph passes must mutate and return the supplied graph")
        graph.validate()
    return graph


PASS_REGISTRY: dict[str, Callable[..., TransformationPass]] = {
    "highlight": HighlightNodes,
    "hide-constants": HideConstants,
    "hide-constant-converts": HideConstantConverts,
}


def register_pass(name: str, factory: Callable[..., TransformationPass]) -> None:
    """Add a named pass factory without changing loading or rendering code."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Pass names must be nonempty strings")
    if name in PASS_REGISTRY:
        raise ValueError(f"Pass {name!r} is already registered")
    if not callable(factory):
        raise ValueError("A pass factory must be callable")
    PASS_REGISTRY[name] = factory


def create_pass(name: str, **options: object) -> TransformationPass:
    try:
        factory = PASS_REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(PASS_REGISTRY))
        raise ValueError(f"Unknown pass {name!r}; available passes: {available}") from None
    return factory(**options)
