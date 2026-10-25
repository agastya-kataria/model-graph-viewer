import json

from model_graph_viewer import Edge, Graph, HideConstants, Node, TensorMetadata, apply_passes
from model_graph_viewer.metadata import edge_counts, metadata_path_for, snapshot_graph
from model_graph_viewer.render import render_svg


def test_snapshot_is_detached_and_preserves_shape_distinctions():
    graph = Graph(
        [
            Node(
                -3,
                "Custom",
                "duplicate",
                attributes={"axes": [1, {"bounds": [2, 8]}]},
                input_ports={0: TensorMetadata(None), 1: TensorMetadata("?")},
                output_ports={2: TensorMetadata(()), 4: TensorMetadata(("2..8", "?"), "f32")},
            )
        ]
    )
    snapshot = snapshot_graph(graph)
    node = snapshot["nodes"]["-3"]
    assert node["input_ports"]["0"]["shape"] is None
    assert node["input_ports"]["1"]["shape"] == "?"
    assert node["output_ports"]["2"]["shape"] == []
    assert node["output_ports"]["4"]["shape"] == ["2..8", "?"]
    graph.nodes[-3].attributes["axes"][1]["bounds"].append(9)
    graph.nodes[-3].visual.reference_outputs.add(2)
    assert node["attributes"] == {"axes": [1, {"bounds": [2, 8]}]}
    assert node["visual"]["reference_outputs"] == []
    json.dumps(snapshot, allow_nan=False)


def test_original_snapshot_retains_removed_constants_and_topology(tmp_path):
    tensor = TensorMetadata((4,), "f32")
    graph = Graph(
        [
            Node(0, "Constant", "weights", output_ports={0: tensor}),
            Node(1, "Add", "sum", input_ports={0: tensor}),
        ],
        [Edge(0, 0, 1, 0, tensor)],
    )
    original = snapshot_graph(graph)
    apply_passes(graph, [HideConstants()])
    output = render_svg(graph, tmp_path / "graph.svg", _original=original)
    data = json.loads(output.with_suffix(".json").read_text())
    assert data["schema_version"] == 1
    assert set(data["nodes"]) == {"0", "1"}
    assert data["nodes"]["0"]["name"] == "weights"
    assert data["edges"] == original["edges"]
    assert set(data["view"]["nodes"]) == {"1"}
    assert data["view"]["edges"] == []
    hidden = data["view"]["nodes"]["1"]["visual"]["hidden_inputs"]["0"]
    assert hidden["source_id"] == 0
    assert hidden["tensor"] == {"shape": [4], "dtype": "f32"}


def test_edge_counts_classifies_each_source_output_independently():
    graph = Graph(
        [
            Node(0, "Split", output_ports={0: TensorMetadata(), 1: TensorMetadata()}),
            Node(1, "Join", input_ports={0: TensorMetadata(), 1: TensorMetadata()}),
        ],
        [Edge(0, 0, 1, 0), Edge(0, 1, 1, 1)],
    )
    assert edge_counts(graph) == (2, 0)
    graph.nodes[0].visual.reference_outputs.add(1)
    assert edge_counts(graph) == (1, 1)


def test_metadata_default_and_explicit_paths_expand_user(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert metadata_path_for("nested/graph.after.svg").as_posix() == "nested/graph.after.json"
    assert metadata_path_for("out.svg", "~/detail.json") == tmp_path / "detail.json"
