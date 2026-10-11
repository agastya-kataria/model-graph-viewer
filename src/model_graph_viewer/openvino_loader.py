"""Read OpenVINO IR into a detached, framework-independent graph.

Only this adapter knows about OpenVINO. Reading a model never compiles it or
copies Constant tensor contents into the visualization graph.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from .graph import Edge, Graph, Node, TensorMetadata


class ModelLoadError(ValueError):
    """The supplied model cannot be read as an OpenVINO IR."""


def _portable(value: Any) -> object:
    """Detach attribute values without retaining framework-owned objects."""
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
    # Some non-Constant operation attributes are NumPy arrays/scalars. Only
    # attribute values reach here; ports and Constant data never do.
    if type(value).__module__.startswith("numpy") and hasattr(value, "tolist"):
        return _portable(value.tolist())
    return str(value)


def _shape(partial_shape: Any) -> tuple[int | str, ...] | str:
    if partial_shape.rank.is_dynamic:
        return "?"
    dimensions: list[int | str] = []
    for dimension in partial_shape:
        if dimension.is_static:
            dimensions.append(int(dimension.get_length()))
            continue
        minimum = int(dimension.get_min_length())
        maximum = int(dimension.get_max_length())
        if minimum == 0 and maximum == -1:
            dimensions.append("?")
        else:
            upper = "?" if maximum == -1 else str(maximum)
            dimensions.append(f"{minimum}..{upper}")
    return tuple(dimensions)


def _tensor(port: Any) -> TensorMetadata:
    return TensorMetadata(
        shape=_shape(port.get_partial_shape()),
        dtype=port.get_element_type().get_type_name(),
    )


class OpenVINOLoader:
    """Load an IR or adapt an existing OpenVINO model without modifying it."""

    def load(self, path: str | Path) -> Graph:
        """Read ``path`` and its adjacent weights, if the IR requires them.

        An IR without Constants may legitimately have no ``.bin`` file. Let
        OpenVINO decide whether weights are required instead of rejecting it.
        """
        model_path = Path(path).expanduser()
        if model_path.suffix.lower() != ".xml":
            raise ModelLoadError(f"Expected an OpenVINO .xml model: {model_path}")
        if not model_path.is_file():
            raise ModelLoadError(f"Model file does not exist: {model_path}")
        try:
            from openvino import Core
        except ImportError as exc:
            raise ModelLoadError(
                "OpenVINO is unavailable. Install the project's OpenVINO dependency."
            ) from exc
        try:
            model = Core().read_model(model=str(model_path))
            return self.from_model(model)
        except (RuntimeError, ValueError, TypeError, OSError) as exc:
            raise ModelLoadError(
                f"Could not read OpenVINO model '{model_path}'. Check that the XML "
                "is valid and that any required weights are in the matching .bin "
                f"file beside it. OpenVINO reported: {exc}"
            ) from exc

    def from_model(self, model: Any) -> Graph:
        """Copy topology, port metadata and attributes from an OpenVINO model.

        ``get_ordered_ops`` includes stateful sinks. Instance IDs identify
        operations while adapting; consecutive IDs identify the detached graph.
        Friendly names are labels and need not be unique. Port indices are the
        runtime zero-based positions, not the IDs written into serialized IR.
        """
        operations = model.get_ordered_ops()
        node_ids = {
            operation.get_instance_id(): node_id for node_id, operation in enumerate(operations)
        }
        nodes: list[Node] = []
        edges: list[Edge] = []
        for node_id, operation in enumerate(operations):
            node_type = operation.get_type_name()
            # Constant.get_attributes() can expose tensor contents as a value
            # attribute. Shape and dtype already live on the output port.
            attributes = (
                {}
                if node_type == "Constant"
                else {
                    str(key): _portable(value) for key, value in operation.get_attributes().items()
                }
            )
            input_ports = {int(port.get_index()): _tensor(port) for port in operation.inputs()}
            output_ports = {int(port.get_index()): _tensor(port) for port in operation.outputs()}
            nodes.append(
                Node(
                    node_id=node_id,
                    node_type=node_type,
                    name=operation.get_friendly_name(),
                    attributes=attributes,
                    input_ports=input_ports,
                    output_ports=output_ports,
                )
            )
            for port in operation.inputs():
                source = port.get_source_output()
                edges.append(
                    Edge(
                        from_node=node_ids[source.get_node().get_instance_id()],
                        from_port=int(source.get_index()),
                        to_node=node_id,
                        to_port=int(port.get_index()),
                        tensor=input_ports[int(port.get_index())],
                    )
                )
        graph = Graph(nodes=nodes, edges=edges)
        graph.validate()
        return graph
