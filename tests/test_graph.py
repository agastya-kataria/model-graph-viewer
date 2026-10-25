from dataclasses import FrozenInstanceError

import pytest

from model_graph_viewer.graph import (
    ConstantAnnotation,
    Edge,
    Graph,
    Node,
    TensorMetadata,
)


def node(node_id, *, inputs=(), outputs=(), name="same", node_type="Custom"):
    tensor = TensorMetadata(("?", 4), "f32")
    return Node(
        node_id,
        node_type,
        name=name,
        input_ports={index: tensor for index in inputs},
        output_ports={index: tensor for index in outputs},
    )


def test_duplicate_names_and_parallel_connections_have_independent_identity():
    left = node(4, outputs=(0, 3))
    right = node(9, inputs=(1, 7))
    edges = [Edge(4, 0, 9, 1), Edge(4, 3, 9, 7)]
    graph = Graph([left, right], edges)

    graph.validate()
    assert len(graph.nodes) == 2
    assert list(graph.edges) == edges
    assert list(graph.outgoing(4)) == edges
    assert list(graph.incoming(9)) == edges
    assert graph.nodes[4].name == graph.nodes[9].name


def test_connection_and_node_removal_update_indexes_and_retain_port_tables():
    source = node(0, outputs=(0, 1))
    consumer = node(1, inputs=(0, 2), outputs=(0,))
    final = node(2, inputs=(0,))
    a, b, c = Edge(0, 0, 1, 0), Edge(0, 1, 1, 2), Edge(1, 0, 2, 0)
    graph = Graph([source, consumer, final], [a, b, c])

    graph.remove_edge(a)
    graph.validate()
    assert list(graph.incoming(1)) == [b]
    assert set(consumer.input_ports) == {0, 2}
    graph.add_edge(a)
    graph.remove_node(1)
    graph.validate()

    assert set(graph.nodes) == {0, 2}
    assert not graph.edges
    assert set(final.input_ports) == {0}
    assert set(source.output_ports) == {0, 1}


def test_self_loop_can_be_removed_without_double_deletion():
    loop = node(0, inputs=(0,), outputs=(0,))
    graph = Graph([loop], [Edge(0, 0, 0, 0)])
    graph.validate()
    graph.remove_node(0)
    graph.validate()
    assert not graph.nodes
    assert not graph.edges


@pytest.mark.parametrize("shape", [None, "?", (), ("?", 8), ("1..8",), ("3..?",), ("0..4",)])
def test_shape_forms_survive_graph_copy(shape):
    tensor = TensorMetadata(shape, "f16")
    graph = Graph(
        [
            Node(0, "Parameter", output_ports={0: tensor}),
            Node(1, "Result", input_ports={0: tensor}),
        ],
        [Edge(0, 0, 1, 0, tensor)],
    )
    duplicate = graph.copy()
    duplicate.validate()
    assert duplicate.nodes[0].output_ports[0].shape == shape
    assert next(iter(duplicate.edges)).tensor.shape == shape


def test_copy_detaches_all_mutable_node_metadata():
    original = node(0, inputs=(3,), outputs=(2,))
    original.attributes["nested"] = {"axes": [1, 2]}
    original.visual.hidden_inputs[3] = ConstantAnnotation(9, TensorMetadata((), "i64"))
    original.visual.reference_outputs.add(2)
    graph = Graph([original])
    duplicate = graph.copy()
    copied = duplicate.nodes[0]
    copied.name = "changed"
    copied.attributes["nested"]["axes"].append(3)
    copied.visual.border_color = "red"
    copied.visual.hidden_inputs.clear()
    copied.visual.reference_outputs.clear()
    copied.input_ports[5] = TensorMetadata((4,), "f32")

    assert original.name == "same"
    assert original.attributes["nested"] == {"axes": [1, 2]}
    assert original.visual.border_color is None
    assert set(original.visual.hidden_inputs) == {3}
    assert original.visual.reference_outputs == {2}
    assert set(original.input_ports) == {3}
    graph.validate()
    duplicate.validate()


@pytest.mark.parametrize("invalid_port", [True, False, 1.0, -1, "1", None])
def test_reference_outputs_reject_invalid_port_indices(invalid_port):
    source = node(0, outputs=(0, 1))
    source.visual.reference_outputs.add(invalid_port)
    with pytest.raises(ValueError, match="invalid reference output port"):
        Graph([source])


def test_reference_outputs_must_resolve_to_existing_output_ports():
    source = node(-4, outputs=(2,))
    source.visual.reference_outputs.add(2)
    graph = Graph([source])
    graph.validate()

    source.visual.reference_outputs.add(5)
    with pytest.raises(ValueError, match="reference for absent output 5"):
        graph.validate()
    source.visual.reference_outputs.remove(5)
    source.output_ports.clear()
    with pytest.raises(ValueError, match="reference for absent output 2"):
        graph.validate()


def test_collections_and_edge_metadata_are_read_only():
    graph = Graph([node(0, outputs=(0,)), node(1, inputs=(0,))], [Edge(0, 0, 1, 0)])
    with pytest.raises(TypeError):
        graph.nodes[2] = node(2)
    with pytest.raises(AttributeError):
        graph.edges.clear()
    edge = next(iter(graph.edges))
    with pytest.raises(FrozenInstanceError):
        edge.to_port = 8
    with pytest.raises(FrozenInstanceError):
        edge.tensor.dtype = "i32"


def test_disconnected_nodes_and_unknown_operation_types_are_valid():
    graph = Graph([node(0, node_type="VendorOperation"), node(1)])
    graph.validate()
    assert not graph.incoming(0)
    assert not graph.outgoing(0)


@pytest.mark.parametrize(
    "edge, message",
    [
        (Edge(8, 0, 1, 0), "absent node"),
        (Edge(0, 0, 8, 0), "absent node"),
        (Edge(0, 7, 1, 0), "output port"),
        (Edge(0, 0, 1, 7), "input port"),
        (Edge(0, True, 1, 0), "output port"),
        (Edge(False, 0, 1, 0), "IDs must be integers"),
    ],
)
def test_invalid_edge_is_rejected_without_mutating_graph(edge, message):
    graph = Graph([node(0, outputs=(0,)), node(1, inputs=(0,))])
    with pytest.raises(ValueError, match=message):
        graph.add_edge(edge)
    graph.validate()
    assert not graph.edges


def test_duplicate_connections_and_multiple_producers_are_rejected():
    edge = Edge(0, 0, 2, 0)
    graph = Graph(
        [node(0, outputs=(0, 1)), node(1, outputs=(0,)), node(2, inputs=(0,))],
        [edge],
    )
    with pytest.raises(ValueError, match="Duplicate edge"):
        graph.add_edge(edge)
    with pytest.raises(ValueError, match="already has a producer"):
        graph.add_edge(Edge(1, 0, 2, 0))
    with pytest.raises(ValueError, match="already has a producer"):
        graph.add_edge(Edge(0, 1, 2, 0))
    graph.validate()


def test_duplicate_ids_and_invalid_ports_are_rejected():
    with pytest.raises(ValueError, match="Duplicate node ID"):
        Graph([node(0), node(0)])
    with pytest.raises(ValueError, match="invalid input port"):
        Graph([node(0, inputs=(-1,))])
    with pytest.raises(ValueError, match="needs TensorMetadata"):
        Graph([Node(0, "Add", input_ports={0: None})])


def test_validate_detects_external_id_and_port_edits():
    graph = Graph([node(0, outputs=(0,)), node(1, inputs=(0,))], [Edge(0, 0, 1, 0)])
    graph.nodes[0].node_id = 12
    with pytest.raises(ValueError, match="ID changed"):
        graph.validate()
    graph.nodes[0].node_id = 0
    graph.nodes[0].output_ports.clear()
    with pytest.raises(ValueError, match="output port"):
        graph.validate()


def test_hidden_annotation_must_have_port_and_cannot_have_visible_producer():
    consumer = node(1, inputs=(0,))
    annotation = ConstantAnnotation(5, TensorMetadata((), "i64"))
    consumer.visual.hidden_inputs[4] = annotation
    with pytest.raises(ValueError, match="absent input"):
        Graph([consumer])
    consumer.visual.hidden_inputs = {0: annotation}
    graph = Graph([node(0, outputs=(0,)), consumer])
    with pytest.raises(ValueError, match="hidden producer"):
        graph.add_edge(Edge(0, 0, 1, 0))
    graph.validate()


def test_absent_removals_and_queries_fail_clearly():
    graph = Graph()
    with pytest.raises(ValueError, match="Unknown node ID"):
        graph.remove_node(12)
    with pytest.raises(ValueError, match="Unknown node ID"):
        graph.incoming(12)
    with pytest.raises(ValueError, match="Unknown node ID"):
        graph.outgoing(12)
    with pytest.raises(ValueError, match="absent node"):
        graph.remove_edge(Edge(0, 0, 1, 0))


@pytest.mark.parametrize("invalid_id", [True, False, 1.0, 0.0])
def test_queries_and_removal_reject_noninteger_id_aliases(invalid_id):
    graph = Graph([node(0), node(1)])
    for operation in (graph.incoming, graph.outgoing, graph.remove_node):
        with pytest.raises(ValueError, match="IDs must be integers"):
            operation(invalid_id)
    assert set(graph.nodes) == {0, 1}
    graph.validate()


@pytest.mark.parametrize("invalid_port", [True, 1.0])
def test_hidden_annotations_reject_noninteger_port_aliases(invalid_port):
    consumer = node(0, inputs=(1,))
    consumer.visual.hidden_inputs[invalid_port] = ConstantAnnotation(7, TensorMetadata())
    with pytest.raises(ValueError, match="invalid hidden input port"):
        Graph([consumer])


@pytest.mark.parametrize(
    "annotation, message",
    [
        (ConstantAnnotation(None, TensorMetadata()), "annotation IDs"),
        (ConstantAnnotation(True, TensorMetadata()), "annotation IDs"),
        (ConstantAnnotation(7, TensorMetadata(), convert_id=2.0), "annotation IDs"),
        (ConstantAnnotation(7, "not metadata"), "needs TensorMetadata"),
        (ConstantAnnotation(7, TensorMetadata([4], "f32")), "shape must be"),
        (ConstantAnnotation(7, TensorMetadata((4,), 32)), "dtype must be"),
    ],
)
def test_hidden_annotations_validate_identity_and_tensor_metadata(annotation, message):
    consumer = node(0, inputs=(0,))
    consumer.visual.hidden_inputs[0] = annotation
    with pytest.raises(ValueError, match=message):
        Graph([consumer])


@pytest.mark.parametrize("shape", [[4], "4", (False,), (-1,), ([4],)])
def test_invalid_shapes_fail_validation_before_hashing_edges(shape):
    tensor = TensorMetadata(shape, "f32")
    with pytest.raises(ValueError, match="shape must be|dimensions must be"):
        Graph([Node(0, "Parameter", output_ports={0: tensor})])
    graph = Graph([node(0, outputs=(0,)), node(1, inputs=(0,))])
    with pytest.raises(ValueError, match="shape must be|dimensions must be"):
        graph.add_edge(Edge(0, 0, 1, 0, tensor))
    assert not graph.edges
    graph.validate()


def test_negative_integer_node_and_annotation_ids_remain_valid():
    consumer = node(-1, inputs=(0,))
    consumer.visual.hidden_inputs[0] = ConstantAnnotation(-7, TensorMetadata(), convert_id=-4)
    graph = Graph([consumer])
    graph.validate()
    assert not graph.incoming(-1)
    graph.remove_node(-1)
    assert not graph.nodes


@pytest.mark.parametrize(
    "invalid_edge",
    [
        Edge(False, 0, 1, 0, TensorMetadata((1,), "f32")),
        Edge(0, 0, 1.0, 0, TensorMetadata((1,), "f32")),
        Edge(0, False, 1, 0, TensorMetadata((1,), "f32")),
        Edge(0, 0, 1, 0.0, TensorMetadata((1,), "f32")),
        Edge(0, 0, 1, 0, TensorMetadata((True,), "f32")),
        Edge(0, 0, 1, 0, TensorMetadata([1], "f32")),
    ],
)
def test_edge_removal_rejects_invalid_aliases_without_deleting_valid_connection(invalid_edge):
    tensor = TensorMetadata((1,), "f32")
    edge = Edge(0, 0, 1, 0, tensor)
    graph = Graph([node(0, outputs=(0,)), node(1, inputs=(0,))], [edge])

    with pytest.raises(ValueError):
        graph.remove_edge(invalid_edge)

    assert list(graph.edges) == [edge]
    assert list(graph.outgoing(0)) == [edge]
    assert list(graph.incoming(1)) == [edge]
    graph.validate()


def test_removing_absent_edge_between_valid_ports_is_actionable():
    graph = Graph([node(0, outputs=(0,)), node(1, inputs=(0,))])
    with pytest.raises(ValueError, match="absent edge"):
        graph.remove_edge(Edge(0, 0, 1, 0))
    graph.validate()
