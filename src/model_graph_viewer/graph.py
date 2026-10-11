"""Framework-independent graph data and indexed, validated mutations.

Names are display text, never identity. Port indices belong to the operation;
removing a connection does not renumber or delete the operation's ports.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType

Shape = tuple[int | str, ...] | str | None


@dataclass(frozen=True)
class TensorMetadata:
    """Tensor description without values.

    ``None`` means unavailable, ``"?"`` means dynamic rank, ``()`` is a
    scalar, and tuple dimensions may contain ``"?"`` or bounds like ``"1..8"``.
    """

    shape: Shape = None
    dtype: str | None = None


@dataclass(frozen=True)
class ConstantAnnotation:
    """An input's hidden constant, optionally viewed through a Convert."""

    source_id: int
    tensor: TensorMetadata
    shared: bool = False
    convert_id: int | None = None
    stored_dtype: str | None = None
    source_shared: bool = False

    @property
    def identity(self) -> int:
        """Identity of the effective value, rather than its display name."""
        return self.source_id if self.convert_id is None else self.convert_id


@dataclass
class Visualization:
    border_color: str | None = None
    hidden_inputs: dict[int, ConstantAnnotation] = field(default_factory=dict)
    # Retain sharing provenance when a previous pass removes a consumer.
    constant_shared: bool = False


@dataclass
class Node:
    node_id: int
    node_type: str
    name: str = ""
    attributes: dict[str, object] = field(default_factory=dict)
    input_ports: dict[int, TensorMetadata] = field(default_factory=dict)
    output_ports: dict[int, TensorMetadata] = field(default_factory=dict)
    visual: Visualization = field(default_factory=Visualization)


@dataclass(frozen=True)
class Edge:
    from_node: int
    from_port: int
    to_node: int
    to_port: int
    tensor: TensorMetadata = field(default_factory=TensorMetadata)


class Graph:
    """An insertion-ordered graph with constant-time connection mutations.

    Collections are read-only views. Nodes and their metadata remain mutable;
    use the graph's mutation methods for connectivity and ``validate`` after
    editing port definitions. Construction and validation take O(V + E).
    """

    def __init__(self, nodes: Iterable[Node] = (), edges: Iterable[Edge] = ()) -> None:
        self._nodes: dict[int, Node] = {}
        self._edges: dict[Edge, None] = {}
        self._incoming: dict[int, dict[Edge, None]] = {}
        self._outgoing: dict[int, dict[Edge, None]] = {}
        self._producers: dict[tuple[int, int], Edge] = {}
        for node in nodes:
            self.add_node(node)
        for edge in edges:
            self.add_edge(edge)

    @property
    def nodes(self) -> Mapping[int, Node]:
        return MappingProxyType(self._nodes)

    @property
    def edges(self) -> Collection[Edge]:
        return self._edges.keys()

    @staticmethod
    def _validate_tensor(tensor: TensorMetadata, context: str) -> None:
        if not isinstance(tensor, TensorMetadata):
            raise ValueError(f"{context} needs TensorMetadata")
        if tensor.dtype is not None and not isinstance(tensor.dtype, str):
            raise ValueError(f"{context} dtype must be a string or None")
        shape = tensor.shape
        if shape is None or (isinstance(shape, str) and shape == "?"):
            return
        if not isinstance(shape, tuple):
            raise ValueError(f"{context} shape must be a tuple, '?', or None")
        for dimension in shape:
            if isinstance(dimension, str):
                continue
            if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 0:
                raise ValueError(f"{context} dimensions must be nonnegative integers or strings")

    @staticmethod
    def _validate_node(node: Node) -> None:
        if not isinstance(node.node_id, int) or isinstance(node.node_id, bool):
            raise ValueError("Node IDs must be integers")
        for direction, ports in (
            ("input", node.input_ports),
            ("output", node.output_ports),
        ):
            for index, tensor in ports.items():
                if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                    raise ValueError(f"Node {node.node_id} has invalid {direction} port {index!r}")
                Graph._validate_tensor(tensor, f"Node {node.node_id} {direction} port {index}")
        for index, annotation in node.visual.hidden_inputs.items():
            if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                raise ValueError(f"Node {node.node_id} has invalid hidden input port {index!r}")
            if index not in node.input_ports:
                raise ValueError(f"Node {node.node_id} has an annotation for absent input {index}")
            if not isinstance(annotation, ConstantAnnotation):
                raise ValueError(f"Node {node.node_id} input {index} needs ConstantAnnotation")
            identities = (annotation.source_id,)
            if annotation.convert_id is not None:
                identities += (annotation.convert_id,)
            for identity in identities:
                if not isinstance(identity, int) or isinstance(identity, bool):
                    raise ValueError("Constant annotation IDs must be integers")
            Graph._validate_tensor(annotation.tensor, f"Node {node.node_id} hidden input {index}")

    def _check_endpoint(self, node_id: int, port: int, *, output: bool) -> None:
        if not isinstance(node_id, int) or isinstance(node_id, bool):
            raise ValueError("Edge node IDs must be integers")
        if node_id not in self._nodes:
            raise ValueError(f"Edge references absent node {node_id}")
        node = self._nodes[node_id]
        ports = node.output_ports if output else node.input_ports
        if not isinstance(port, int) or isinstance(port, bool) or port not in ports:
            direction = "output" if output else "input"
            raise ValueError(f"Node {node_id} has no {direction} port {port}")

    def add_node(self, node: Node) -> None:
        self._validate_node(node)
        if node.node_id in self._nodes:
            raise ValueError(f"Duplicate node ID {node.node_id}")
        self._nodes[node.node_id] = node
        self._incoming[node.node_id] = {}
        self._outgoing[node.node_id] = {}

    def add_edge(self, edge: Edge) -> None:
        self._check_endpoint(edge.from_node, edge.from_port, output=True)
        self._check_endpoint(edge.to_node, edge.to_port, output=False)
        self._validate_tensor(edge.tensor, "Edge")
        if edge in self._edges:
            raise ValueError(f"Duplicate edge {edge}")
        destination = (edge.to_node, edge.to_port)
        if destination in self._producers:
            raise ValueError(f"Node {edge.to_node} input {edge.to_port} already has a producer")
        if edge.to_port in self._nodes[edge.to_node].visual.hidden_inputs:
            raise ValueError(
                f"Node {edge.to_node} input {edge.to_port} already has a hidden producer"
            )
        self._edges[edge] = None
        self._incoming[edge.to_node][edge] = None
        self._outgoing[edge.from_node][edge] = None
        self._producers[destination] = edge

    def remove_edge(self, edge: Edge) -> None:
        self._check_endpoint(edge.from_node, edge.from_port, output=True)
        self._check_endpoint(edge.to_node, edge.to_port, output=False)
        self._validate_tensor(edge.tensor, "Edge")
        if edge not in self._edges:
            raise ValueError(f"Cannot remove absent edge {edge}")
        del self._edges[edge]
        del self._incoming[edge.to_node][edge]
        del self._outgoing[edge.from_node][edge]
        del self._producers[(edge.to_node, edge.to_port)]

    def remove_node(self, node_id: int) -> None:
        """Remove a node and its incident edges; retain every other port table."""
        self._require_node(node_id)
        # A self-loop appears in both indexes, so remove outgoing edges second.
        for edge in tuple(self._incoming[node_id]):
            self.remove_edge(edge)
        for edge in tuple(self._outgoing[node_id]):
            self.remove_edge(edge)
        del self._incoming[node_id]
        del self._outgoing[node_id]
        del self._nodes[node_id]

    def _require_node(self, node_id: int) -> None:
        if not isinstance(node_id, int) or isinstance(node_id, bool):
            raise ValueError("Node IDs must be integers")
        if node_id not in self._nodes:
            raise ValueError(f"Unknown node ID {node_id}")

    def incoming(self, node_id: int) -> Collection[Edge]:
        self._require_node(node_id)
        return self._incoming[node_id].keys()

    def outgoing(self, node_id: int) -> Collection[Edge]:
        self._require_node(node_id)
        return self._outgoing[node_id].keys()

    def copy(self) -> Graph:
        """Copy all mutable node data for an independent visualization variant."""
        return Graph(deepcopy(tuple(self._nodes.values())), self._edges)

    def validate(self) -> None:
        """Check endpoints, metadata, single producers, and adjacency indexes."""
        if self._incoming.keys() != self._nodes.keys():
            raise ValueError("Incoming adjacency index has inconsistent nodes")
        if self._outgoing.keys() != self._nodes.keys():
            raise ValueError("Outgoing adjacency index has inconsistent nodes")
        for node_id, node in self._nodes.items():
            if node_id != node.node_id:
                raise ValueError(f"Node ID changed after insertion: {node_id}")
            self._validate_node(node)
        if (
            len(self._producers) != len(self._edges)
            or sum(map(len, self._incoming.values())) != len(self._edges)
            or sum(map(len, self._outgoing.values())) != len(self._edges)
        ):
            raise ValueError("Adjacency index has inconsistent edge counts")
        for edge in self._edges:
            self._check_endpoint(edge.from_node, edge.from_port, output=True)
            self._check_endpoint(edge.to_node, edge.to_port, output=False)
            self._validate_tensor(edge.tensor, "Edge")
            if edge.to_port in self._nodes[edge.to_node].visual.hidden_inputs:
                raise ValueError(
                    f"Node {edge.to_node} input {edge.to_port} has visible and hidden producers"
                )
            if (
                edge not in self._incoming[edge.to_node]
                or edge not in self._outgoing[edge.from_node]
                or self._producers.get((edge.to_node, edge.to_port)) != edge
            ):
                raise ValueError("Adjacency index has inconsistent edges")
