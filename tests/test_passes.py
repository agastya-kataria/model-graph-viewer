from dataclasses import asdict

import pytest

from model_graph_viewer.graph import Edge, Graph, Node, TensorMetadata
from model_graph_viewer.passes import (
    PASS_REGISTRY,
    HideConstantConverts,
    HideConstants,
    HighlightNodes,
    apply_passes,
    create_pass,
    register_pass,
)

F16 = TensorMetadata((4, 4), "f16")
F32 = TensorMetadata((4, 4), "f32")
I32 = TensorMetadata((4, 4), "i32")


def node(node_id, op="Custom", *, inputs=(), outputs=(0,), tensor=F32, name="same"):
    return Node(
        node_id,
        op,
        name=name,
        input_ports={port: tensor for port in inputs},
        output_ports={port: tensor for port in outputs},
    )


def test_highlight_exact_matches_only_and_preserves_other_metadata():
    nodes = [node(0, name="target"), node(1, name="target"), node(2, name="target/part")]
    nodes[0].attributes["border_color"] = "an operation attribute"
    nodes[2].visual.border_color = "blue"
    graph = Graph(nodes)
    before = graph.copy()

    result = apply_passes(graph, [HighlightNodes(["target", "absent"])])

    assert result is graph
    assert [item.visual.border_color for item in nodes] == ["red", "red", "blue"]
    for index in (0, 1):
        expected = asdict(before.nodes[index])
        expected["visual"]["border_color"] = "red"
        assert asdict(graph.nodes[index]) == expected
    assert graph.nodes[2] == before.nodes[2]


def test_hide_constants_preserves_nonconstant_connections_and_original_input_indices():
    constant = node(0, "Constant", tensor=F16)
    data = node(1, "Parameter")
    consumer = node(2, "Add", inputs=(0, 7))
    consumer.attributes["mode"] = "numpy"
    consumer.visual.border_color = "green"
    graph = Graph([constant, data, consumer], [Edge(0, 0, 2, 7, F16), Edge(1, 0, 2, 0, F32)])

    apply_passes(graph, [HideConstants()])

    assert set(graph.nodes) == {1, 2}
    assert list(graph.edges) == [Edge(1, 0, 2, 0, F32)]
    assert set(consumer.input_ports) == {0, 7}
    annotation = consumer.visual.hidden_inputs[7]
    assert annotation.source_id == 0
    assert annotation.tensor == F16
    assert not annotation.shared
    assert annotation.convert_id is None
    assert consumer.attributes == {"mode": "numpy"}
    assert consumer.visual.border_color == "green"


def test_multiple_uses_within_one_consumer_do_not_count_as_sharing():
    graph = Graph(
        [node(0, "Constant"), node(1, "Add", inputs=(0, 1))],
        [Edge(0, 0, 1, 0), Edge(0, 0, 1, 1)],
    )
    apply_passes(graph, [HideConstants()])
    annotations = graph.nodes[1].visual.hidden_inputs
    assert set(annotations) == {0, 1}
    assert {item.source_id for item in annotations.values()} == {0}
    assert not any(item.shared for item in annotations.values())


def test_shared_constants_keep_source_identity_without_using_duplicate_names():
    graph = Graph(
        [node(0, "Constant"), node(1, "Constant"), node(2, inputs=(0, 1)), node(3, inputs=(0,))],
        [Edge(0, 0, 2, 0), Edge(1, 0, 2, 1), Edge(0, 0, 3, 0)],
    )
    apply_passes(graph, [HideConstants()])
    shared = graph.nodes[2].visual.hidden_inputs[0]
    other = graph.nodes[2].visual.hidden_inputs[1]
    assert shared.shared
    assert shared.identity == 0
    assert graph.nodes[3].visual.hidden_inputs[0] == shared
    assert not other.shared
    assert other.identity == 1


def test_hide_constants_removes_orphans_and_never_reads_tensor_values():
    class NoTensorValues:
        def __iter__(self):
            raise AssertionError("Tensor values were accessed")

    constant = node(0, "Constant", tensor=TensorMetadata((), "i64"))
    constant.attributes["opaque"] = NoTensorValues()
    graph = Graph([constant, node(1, inputs=(3,)), node(2, "Constant")], [Edge(0, 0, 1, 3)])
    apply_passes(graph, [HideConstants()])
    assert set(graph.nodes) == {1}
    assert graph.nodes[1].visual.hidden_inputs[3].tensor == TensorMetadata((), "i64")
    assert set(asdict(graph.nodes[1].visual.hidden_inputs[3])) == {
        "source_id",
        "tensor",
        "shared",
        "convert_id",
        "stored_dtype",
        "source_shared",
    }


def converted_branch_graph():
    """One storage value, two dtypes, a shared Convert, and a direct consumer."""
    return Graph(
        [
            node(0, "Constant", tensor=F16),
            node(1, "Convert", inputs=(0,), tensor=F32),
            node(2, "Convert", inputs=(0,), tensor=I32),
            node(3, inputs=(0, 4)),
            node(4, inputs=(2,)),
            node(5, inputs=(0,)),
            node(6, "Parameter"),
        ],
        [
            Edge(0, 0, 1, 0, F16),
            Edge(0, 0, 2, 0, F16),
            Edge(0, 0, 3, 0, F16),
            Edge(1, 0, 3, 4, F32),
            Edge(1, 0, 4, 2, F32),
            Edge(2, 0, 5, 0, I32),
        ],
    )


def test_optional_pass_distinguishes_storage_converted_identity_and_output_types():
    graph = converted_branch_graph()
    apply_passes(graph, [HideConstantConverts()])

    assert set(graph.nodes) == {0, 3, 4, 5, 6}
    assert list(graph.edges) == [Edge(0, 0, 3, 0, F16)]
    fp = graph.nodes[3].visual.hidden_inputs[4]
    assert fp == graph.nodes[4].visual.hidden_inputs[2]
    assert fp.tensor == F32
    assert fp.stored_dtype == "f16"
    assert fp.source_id == 0
    assert fp.convert_id == fp.identity == 1
    assert fp.shared and fp.source_shared
    integer = graph.nodes[5].visual.hidden_inputs[0]
    assert integer.tensor == I32
    assert integer.stored_dtype == "f16"
    assert integer.convert_id == integer.identity == 2
    assert not integer.shared
    assert integer.source_shared
    assert graph.nodes[0].visual.constant_shared


def test_original_source_sharing_survives_removal_of_converted_branches():
    graph = converted_branch_graph()
    apply_passes(graph, [HideConstantConverts(), HideConstants()])
    direct = graph.nodes[3].visual.hidden_inputs[0]
    assert direct.shared and direct.source_shared
    assert direct.source_id == 0
    assert direct.convert_id is None
    assert direct.tensor == F16
    assert set(graph.nodes) == {3, 4, 5, 6}
    assert not graph.edges


def test_reversed_order_leaves_convert_nodes_and_their_outputs_visible():
    graph = converted_branch_graph()
    apply_passes(graph, [HideConstants()])
    expected = graph.copy()
    apply_passes(graph, [HideConstantConverts()])
    assert graph.nodes == expected.nodes
    assert list(graph.edges) == list(expected.edges)
    assert graph.nodes[1].visual.hidden_inputs[0].tensor == F16
    assert graph.nodes[1].node_type == "Convert"


def test_longer_conversion_chain_remains_partially_visible():
    graph = Graph(
        [
            node(0, "Constant", tensor=F16),
            node(1, "Convert", inputs=(0,), tensor=F32),
            node(2, "Convert", inputs=(0,), tensor=I32),
            node(3, inputs=(0,)),
        ],
        [Edge(0, 0, 1, 0, F16), Edge(1, 0, 2, 0, F32), Edge(2, 0, 3, 0, I32)],
    )
    apply_passes(graph, [HideConstantConverts(), HideConstants()])
    assert set(graph.nodes) == {2, 3}
    assert list(graph.edges) == [Edge(2, 0, 3, 0, I32)]
    assert graph.nodes[2].visual.hidden_inputs[0].convert_id == 1


@pytest.mark.parametrize("inputs, outputs", [((0, 1), (0,)), ((0,), (0, 1))])
def test_optional_pass_does_not_fold_other_convert_signatures(inputs, outputs):
    graph = Graph(
        [
            node(0, "Constant"),
            node(1, "Convert", inputs=inputs, outputs=outputs),
            node(2, inputs=(0,)),
        ],
        [Edge(0, 0, 1, 0), Edge(1, 0, 2, 0)],
    )
    expected = graph.copy()
    apply_passes(graph, [HideConstantConverts()])
    assert graph.nodes == expected.nodes
    assert list(graph.edges) == list(expected.edges)


def test_optional_pass_does_not_fold_nonconstant_inputs():
    graph = Graph(
        [node(0, "Parameter"), node(1, "Convert", inputs=(0,)), node(2, inputs=(0,))],
        [Edge(0, 0, 1, 0), Edge(1, 0, 2, 0)],
    )
    expected = graph.copy()
    apply_passes(graph, [HideConstantConverts()])
    assert graph.nodes == expected.nodes
    assert list(graph.edges) == list(expected.edges)


def test_repeated_converted_value_in_one_operation_is_not_shared():
    graph = Graph(
        [node(0, "Constant", tensor=F16), node(1, "Convert", inputs=(0,)), node(2, inputs=(0, 1))],
        [Edge(0, 0, 1, 0, F16), Edge(1, 0, 2, 0, F32), Edge(1, 0, 2, 1, F32)],
    )
    apply_passes(graph, [HideConstantConverts()])
    annotations = graph.nodes[2].visual.hidden_inputs
    assert annotations[0] == annotations[1]
    assert not annotations[0].shared
    assert not annotations[0].source_shared
    assert annotations[0].convert_id == 1


@pytest.mark.parametrize(
    "pass_types", [(HideConstants,), (HideConstantConverts,), (HideConstantConverts, HideConstants)]
)
def test_hiding_passes_are_idempotent(pass_types):
    graph = converted_branch_graph()
    apply_passes(graph, [factory() for factory in pass_types])
    expected = graph.copy()
    apply_passes(graph, [factory() for factory in pass_types])
    assert graph.nodes == expected.nodes
    assert list(graph.edges) == list(expected.edges)


def test_passes_run_exactly_in_supplied_order_and_registry_is_extensible(monkeypatch):
    class AppendName:
        def __init__(self, suffix):
            self.suffix = suffix

        def apply(self, graph):
            for current in graph.nodes.values():
                current.name += self.suffix
            return graph

    monkeypatch.setattr("model_graph_viewer.passes.PASS_REGISTRY", dict(PASS_REGISTRY))
    register_pass("append-name", AppendName)
    graph = Graph([node(0, name="value")])
    result = apply_passes(
        graph,
        [
            create_pass("append-name", suffix="/a"),
            create_pass("highlight", names=["value/a"]),
            AppendName("/b"),
        ],
    )
    assert result is graph
    assert graph.nodes[0].name == "value/a/b"
    assert graph.nodes[0].visual.border_color == "red"
    with pytest.raises(ValueError, match="already registered"):
        register_pass("append-name", AppendName)
    with pytest.raises(ValueError, match="Unknown pass"):
        create_pass("missing")


def test_pipeline_rejects_invalid_custom_mutations_and_replacement_graphs():
    class ChangeId:
        def apply(self, graph):
            graph.nodes[0].node_id = 9
            return graph

    with pytest.raises(ValueError, match="ID changed"):
        apply_passes(Graph([node(0)]), [ChangeId()])

    class ReplaceGraph:
        def apply(self, graph):
            return graph.copy()

    with pytest.raises(ValueError, match="supplied graph"):
        apply_passes(Graph(), [ReplaceGraph()])


def test_empty_pass_sequence_is_identity_and_validation_still_runs():
    graph = Graph([node(0)])
    assert apply_passes(graph, []) is graph
    graph.nodes[0].node_id = 7
    with pytest.raises(ValueError, match="ID changed"):
        apply_passes(graph, [])


@pytest.mark.parametrize("cycle", [False, True])
def test_constant_hiding_prunes_disconnected_constants_independently_of_node_order(cycle):
    source = node(0, "Constant", inputs=(0,) if cycle else ())
    intermediate = node(1, "Constant", inputs=(0,))
    consumer = node(2, "Add", inputs=(0,))
    edges = [Edge(0, 0, 1, 0), Edge(1, 0, 2, 0)]
    if cycle:
        edges.append(Edge(1, 0, 0, 0))
    graph = Graph([intermediate, source, consumer], edges)

    apply_passes(graph, [HideConstants()])

    assert set(graph.nodes) == {2}
    assert not graph.edges
    assert consumer.visual.hidden_inputs[0].source_id == 1
    expected = graph.copy()
    apply_passes(graph, [HideConstants()])
    assert graph.nodes == expected.nodes


def test_constant_with_unrelated_incoming_connection_is_retained():
    graph = Graph(
        [node(0, "Parameter"), node(1, "Constant", inputs=(0,)), node(2, inputs=(0,))],
        [Edge(0, 0, 1, 0), Edge(1, 0, 2, 0)],
    )
    apply_passes(graph, [HideConstants()])
    assert set(graph.nodes) == {0, 1, 2}
    assert list(graph.edges) == [Edge(0, 0, 1, 0)]
    assert graph.nodes[2].visual.hidden_inputs[0].source_id == 1
