"""Readability options preserve model identities and reject destructive paths."""

import json
import os
import re
import xml.etree.ElementTree as ET

import openvino as ov
import pytest
from openvino import opset13 as ops

from model_graph_viewer import Graph, HighlightNodes, Node, load_model, visualize
from model_graph_viewer.cli import main
from model_graph_viewer.passes import ReferenceHighFanout

SVG = {"s": "http://www.w3.org/2000/svg"}
XLINK_TITLE = "{http://www.w3.org/1999/xlink}title"
XLINK_HREF = "{http://www.w3.org/1999/xlink}href"


@pytest.fixture
def fanout_model(tmp_path):
    parameter = ops.parameter([1, 4], ov.Type.f32, name="fanout_source_name")
    branches = [ops.relu(parameter, name=f"branch_name_{index}") for index in range(4)]
    model = ov.Model(branches, [parameter], "four_branches")
    path = tmp_path / "fanout.xml"
    ov.serialize(model, path, path.with_suffix(".bin"))
    return path


def visible_text(element):
    return " ".join("".join(item.itertext()) for item in element.findall(".//s:text", SVG))


def endpoints(record):
    return tuple(record[key] for key in ("from_node", "from_port", "to_node", "to_port"))


def drawn_endpoints(root):
    connections = set()
    for group in root.findall(".//s:g[@class='edge']", SVG):
        anchor = group.find(".//s:a", SVG)
        match = re.fullmatch(r"(-?\d+):out (\d+) → (-?\d+):in (\d+)", anchor.get(XLINK_TITLE))
        assert match is not None
        connections.add(tuple(map(int, match.groups())))
    return connections


def forbidden_load(*args, **kwargs):
    raise AssertionError("Invalid configuration must be rejected before loading the model")


def test_default_compact_visualization_keeps_all_model_and_port_identities(fanout_model, tmp_path):
    graph = load_model(fanout_model)
    expected_edges = {
        (edge.from_node, edge.from_port, edge.to_node, edge.to_port) for edge in graph.edges
    }
    output = tmp_path / "compact.svg"

    result = visualize(fanout_model, output)

    root = ET.parse(output).getroot()
    metadata = json.loads(output.with_suffix(".json").read_text())
    assert (result.before_nodes, result.after_nodes) == (9, 9)
    assert (result.before_edges, result.after_edges) == (8, 8)
    assert (result.rendered_edges, result.referenced_edges) == (8, 0)
    assert result.metadata_path == str(output.with_suffix(".json"))
    assert drawn_endpoints(root) == expected_edges
    assert {endpoints(edge) for edge in metadata["edges"]} == expected_edges
    assert {endpoints(edge) for edge in metadata["view"]["edges"]} == expected_edges
    assert all(edge["representation"] == "edge" for edge in metadata["view"]["edges"])
    assert metadata["nodes"] == metadata["view"]["nodes"]
    assert metadata["rendering"] == {
        "label_detail": "compact",
        "rendered_edges": 8,
        "referenced_edges": 0,
    }
    assert {item.get("id") for item in root.findall(".//s:g[@class='node']", SVG)} == {
        f"node-{node_id}" for node_id in graph.nodes
    }
    text = visible_text(root)
    for node_id, node in graph.nodes.items():
        record = metadata["nodes"][str(node_id)]
        rendered = root.find(f".//s:g[@id='node-{node_id}']", SVG)
        assert record["name"] == node.name
        assert record["node_type"] == node.node_type
        assert set(record["input_ports"]) == {str(port) for port in node.input_ports}
        assert set(record["output_ports"]) == {str(port) for port in node.output_ports}
        for direction, ports in (("in", node.input_ports), ("out", node.output_ports)):
            for port in ports:
                assert f"{direction} {port}" in visible_text(rendered)
        assert node.name not in text


@pytest.mark.parametrize("threshold, references", [(3, 4), (4, 0)])
def test_api_fanout_references_resolve_every_suppressed_connection(
    threshold, references, fanout_model, tmp_path
):
    output = tmp_path / f"references-{threshold}.svg"
    result = visualize(fanout_model, output, [ReferenceHighFanout(threshold)])
    root = ET.parse(output).getroot()
    metadata = json.loads(output.with_suffix(".json").read_text())
    referenced = {
        endpoints(edge)
        for edge in metadata["view"]["edges"]
        if edge["representation"] == "reference"
    }
    original = {endpoints(edge) for edge in metadata["edges"]}

    assert result.after_nodes == result.before_nodes == 9
    assert result.after_edges == result.before_edges == 8
    assert result.referenced_edges == len(referenced) == references
    assert result.rendered_edges == 8 - references
    assert drawn_endpoints(root) == original - referenced
    for source, output_port, consumer, input_port in referenced:
        marker_id = f"a_ref-{source}-{output_port}-{consumer}-{input_port}"
        marker = root.find(f".//s:g[@id='{marker_id}']", SVG)
        assert marker is not None
        anchor = marker.find(".//s:a", SVG)
        assert anchor.get(XLINK_HREF) == f"#node-{source}"
        assert anchor.get(XLINK_TITLE) == f"{source}:out {output_port} → {consumer}:in {input_port}"
        assert f"in {input_port} ← @{source}:{output_port}" in visible_text(marker)
        assert root.find(f".//s:g[@id='a_source-{source}-{output_port}']", SVG) is not None


def test_full_labels_and_highlighting_work_with_custom_metadata(fanout_model, tmp_path):
    graph = load_model(fanout_model)
    source = next(node for node in graph.nodes.values() if node.node_type == "Parameter")
    metadata_path = tmp_path / "metadata" / "connection-map.json"
    output = tmp_path / "full.svg"

    result = visualize(
        fanout_model,
        output,
        [ReferenceHighFanout(), HighlightNodes([source.name])],
        label_detail="full",
        metadata=metadata_path,
    )

    assert result.metadata_path == str(metadata_path)
    assert not output.with_suffix(".json").exists()
    metadata = json.loads(metadata_path.read_text())
    assert metadata["rendering"]["label_detail"] == "full"
    assert metadata["nodes"][str(source.node_id)]["visual"]["border_color"] is None
    assert metadata["view"]["nodes"][str(source.node_id)]["visual"]["border_color"] == "red"
    root = ET.parse(output).getroot()
    source_group = root.find(f".//s:g[@id='node-{source.node_id}']", SVG)
    assert source.name in visible_text(source_group)
    assert any(item.get("stroke") == "red" for item in source_group.iter())
    assert len(drawn_endpoints(root)) == 4


def test_original_metadata_is_detached_before_passes_mutate_nested_attributes(tmp_path):
    original = Node(41, "Custom", "original_name", attributes={"nested": {"axes": [1, 2]}})
    graph = Graph([original])

    class Loader:
        def load(self, path):
            return graph

    class ChangeMetadata:
        def apply(self, current):
            current.nodes[41].name = "changed_name"
            current.nodes[41].attributes["nested"]["axes"].append(3)
            return current

    output = tmp_path / "detached.svg"
    result = visualize("custom.model", output, [ChangeMetadata()], loader=Loader())
    metadata = json.loads(output.with_suffix(".json").read_text())
    assert metadata["nodes"]["41"]["name"] == "original_name"
    assert metadata["nodes"]["41"]["attributes"] == {"nested": {"axes": [1, 2]}}
    assert metadata["view"]["nodes"]["41"]["name"] == "changed_name"
    assert metadata["view"]["nodes"]["41"]["attributes"] == {"nested": {"axes": [1, 2, 3]}}
    assert (result.before_nodes, result.after_nodes, result.after_edges) == (1, 1, 0)
    assert (result.rendered_edges, result.referenced_edges) == (0, 0)


@pytest.mark.parametrize("threshold, references", [(3, 4), (4, 0)])
def test_cli_fanout_threshold_and_custom_metadata_reach_pipeline(
    threshold, references, fanout_model, tmp_path
):
    output = tmp_path / "cli.svg"
    metadata = tmp_path / "graph-metadata.json"
    stats = tmp_path / "stats.json"
    assert (
        main(
            [
                "--model",
                str(fanout_model),
                "--output",
                str(output),
                "--passes",
                "reference-high-fanout",
                "--fanout-threshold",
                str(threshold),
                "--metadata",
                str(metadata),
                "--stats",
                str(stats),
            ]
        )
        == 0
    )
    counts = json.loads(stats.read_text())
    assert counts["after_edges"] == 8
    assert counts["rendered_edges"] == 8 - references
    assert counts["referenced_edges"] == references
    assert counts["metadata_path"] == str(metadata)
    assert len(drawn_endpoints(ET.parse(output).getroot())) == 8 - references


@pytest.mark.parametrize(
    "extra",
    [
        ["--passes", "reference-high-fanout", "--fanout-threshold", "0"],
        ["--passes", "reference-high-fanout", "--fanout-threshold", "-1"],
        ["--passes", "reference-high-fanout", "--fanout-threshold", "1.5"],
        ["--fanout-threshold", "2"],
        ["--label-detail", "verbose"],
    ],
)
def test_cli_readability_configuration_fails_before_loading(extra, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("model_graph_viewer.api.load_model", forbidden_load)
    output = tmp_path / "out.svg"
    with pytest.raises(SystemExit) as caught:
        main(["--model", "missing.xml", "--output", str(output), *extra])
    assert caught.value.code == 2
    assert "error:" in capsys.readouterr().err
    assert not output.exists()
    assert not output.with_suffix(".json").exists()


def test_api_invalid_label_detail_fails_before_loading(tmp_path, monkeypatch):
    monkeypatch.setattr("model_graph_viewer.api.load_model", forbidden_load)
    output = tmp_path / "out.svg"
    with pytest.raises(ValueError, match="label_detail"):
        visualize("missing.xml", output, label_detail="verbose")
    assert not output.exists()
    assert not output.with_suffix(".json").exists()


def assert_rejected_metadata(entrypoint, model, output, metadata, capsys):
    if entrypoint == "api":
        with pytest.raises(ValueError):
            visualize(model, output, metadata=metadata)
    else:
        args = ["--model", str(model), "--output", str(output)]
        if metadata is not None:
            args.extend(["--metadata", str(metadata)])
        with pytest.raises(SystemExit) as caught:
            main(args)
        assert caught.value.code == 2
        error = capsys.readouterr().err
        assert "error:" in error and "Traceback" not in error


@pytest.mark.parametrize("entrypoint", ["api", "cli"])
@pytest.mark.parametrize("target_kind", ["xml", "bin", "svg"])
@pytest.mark.parametrize("alias_kind", ["same", "symlink", "hardlink"])
def test_custom_metadata_collisions_preserve_inputs_before_loading(
    entrypoint, target_kind, alias_kind, fanout_model, tmp_path, monkeypatch, capsys
):
    output = tmp_path / "out.svg"
    output.write_bytes(b"existing SVG")
    weights = fanout_model.with_suffix(".bin")
    targets = {"xml": fanout_model, "bin": weights, "svg": output}
    target = targets[target_kind]
    metadata = tmp_path / "metadata.json"
    if alias_kind == "same":
        metadata = target
    elif alias_kind == "symlink":
        metadata.symlink_to(target)
    else:
        os.link(target, metadata)
    before = {path: path.read_bytes() for path in targets.values()}
    monkeypatch.setattr("model_graph_viewer.api.load_model", forbidden_load)

    assert_rejected_metadata(entrypoint, fanout_model, output, metadata, capsys)

    assert {path: path.read_bytes() for path in targets.values()} == before
    assert metadata.read_bytes() == before[target]
    assert not output.with_suffix(".json").exists()


@pytest.mark.parametrize("entrypoint", ["api", "cli"])
@pytest.mark.parametrize("target_kind", ["xml", "bin", "svg"])
@pytest.mark.parametrize("alias_kind", ["symlink", "hardlink"])
def test_default_metadata_alias_collisions_are_rejected_before_loading(
    entrypoint, target_kind, alias_kind, fanout_model, tmp_path, monkeypatch, capsys
):
    output = tmp_path / "out.svg"
    output.write_bytes(b"existing SVG")
    weights = fanout_model.with_suffix(".bin")
    targets = {"xml": fanout_model, "bin": weights, "svg": output}
    target = targets[target_kind]
    metadata = output.with_suffix(".json")
    if alias_kind == "symlink":
        metadata.symlink_to(target)
    else:
        os.link(target, metadata)
    before = {path: path.read_bytes() for path in targets.values()}
    monkeypatch.setattr("model_graph_viewer.api.load_model", forbidden_load)

    assert_rejected_metadata(entrypoint, fanout_model, output, None, capsys)

    assert {path: path.read_bytes() for path in targets.values()} == before
    assert metadata.read_bytes() == before[target]


@pytest.mark.parametrize("custom_metadata", [False, True])
@pytest.mark.parametrize("alias_kind", ["same", "symlink", "hardlink"])
def test_stats_cannot_alias_default_or_custom_metadata(
    custom_metadata, alias_kind, fanout_model, tmp_path, monkeypatch, capsys
):
    output = tmp_path / "out.svg"
    output.write_bytes(b"existing SVG")
    metadata = tmp_path / "custom.json" if custom_metadata else output.with_suffix(".json")
    metadata.write_bytes(b"existing metadata")
    stats = tmp_path / "stats.json"
    if alias_kind == "same":
        stats = metadata
    elif alias_kind == "symlink":
        stats.symlink_to(metadata)
    else:
        os.link(metadata, stats)
    protected = [fanout_model, fanout_model.with_suffix(".bin"), output, metadata]
    before = {path: path.read_bytes() for path in protected}
    args = ["--model", str(fanout_model), "--output", str(output), "--stats", str(stats)]
    if custom_metadata:
        args.extend(["--metadata", str(metadata)])
    monkeypatch.setattr("model_graph_viewer.api.load_model", forbidden_load)

    with pytest.raises(SystemExit) as caught:
        main(args)

    assert caught.value.code == 2
    assert "must be different" in capsys.readouterr().err
    assert {path: path.read_bytes() for path in protected} == before


@pytest.mark.parametrize("entrypoint", ["api", "cli"])
@pytest.mark.parametrize("custom_metadata", [False, True])
def test_metadata_symlink_loop_fails_before_loading(
    entrypoint, custom_metadata, fanout_model, tmp_path, monkeypatch, capsys
):
    output = tmp_path / "out.svg"
    metadata = tmp_path / "loop.json" if custom_metadata else output.with_suffix(".json")
    metadata.symlink_to(metadata.name)
    before = (fanout_model.read_bytes(), fanout_model.with_suffix(".bin").read_bytes())
    monkeypatch.setattr("model_graph_viewer.api.load_model", forbidden_load)

    assert_rejected_metadata(
        entrypoint, fanout_model, output, metadata if custom_metadata else None, capsys
    )

    assert not output.exists()
    assert (fanout_model.read_bytes(), fanout_model.with_suffix(".bin").read_bytes()) == before
