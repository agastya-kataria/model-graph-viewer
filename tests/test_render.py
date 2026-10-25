import json
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from model_graph_viewer import (
    Edge,
    Graph,
    HideConstants,
    HighlightNodes,
    Node,
    RenderError,
    TensorMetadata,
    apply_passes,
    render_svg,
)
from model_graph_viewer.metadata import snapshot_graph
from model_graph_viewer.render import format_tensor, to_dot

TENSOR = TensorMetadata(("?", 4), "f32")


def sample_graph():
    return Graph(
        [
            Node(0, "Split", 'split<&"|{}', output_ports={0: TENSOR, 1: TENSOR}),
            Node(1, "UnfamiliarOperation", "target", input_ports={0: TENSOR, 1: TENSOR}),
        ],
        [Edge(0, 1, 1, 0, TENSOR), Edge(0, 0, 1, 1, TENSOR)],
    )


def test_svg_keeps_parallel_port_connections_special_text_and_highlight(tmp_path):
    graph = sample_graph()
    HighlightNodes(["target"]).apply(graph)
    output = render_svg(graph, tmp_path / "nested" / "graph.svg", label_detail="full")
    root = ET.parse(output).getroot()
    ns = {"s": "http://www.w3.org/2000/svg"}
    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    labels = [element.text for element in root.findall(".//s:text", ns)]
    assert 'split<&"|{}' in labels
    assert "UnfamiliarOperation" in labels
    assert "in 0" in labels and "out 1" in labels
    assert "f32[?, 4]" in labels
    assert len(root.findall(".//s:g[@class='edge']", ns)) == 2
    highlighted = root.find(".//s:g[@id='node-1']", ns)
    assert highlighted is not None
    assert any(element.get("stroke") == "red" for element in highlighted.iter())
    dot = to_dot(graph).source
    assert "n0:out_1:s -> n1:in_0:n" in dot
    assert "n0:out_0:s -> n1:in_1:n" in dot
    assert "strict digraph" not in dot
    assert set(output.parent.iterdir()) == {output, output.with_suffix(".json")}


def test_hidden_constant_attributes_appear_without_values(tmp_path):
    graph = Graph(
        [
            Node(0, "Constant", "weights", output_ports={0: TENSOR}),
            Node(1, "MatMul", "projection", input_ports={3: TENSOR}),
            Node(2, "MatMul", "other", input_ports={1: TENSOR}),
        ],
        [Edge(0, 0, 1, 3, TENSOR), Edge(0, 0, 2, 1, TENSOR)],
    )
    apply_passes(graph, [HideConstants(), HighlightNodes(["projection"])])
    root = ET.parse(render_svg(graph, tmp_path / "hidden.svg")).getroot()
    text = " ".join(root.itertext())
    assert "[3]: f32[?, 4] (shared: 0)" in text
    assert "[1]: f32[?, 4] (shared: 0)" in text
    assert "weights" not in text


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        (None, "[unknown shape]"),
        ("?", "[dynamic rank]"),
        ((), "[]"),
        (("1..8", "?", 2), "[1..8, ?, 2]"),
    ],
)
def test_shape_display_preserves_distinctions(shape, expected):
    assert format_tensor(TensorMetadata(shape)) == "unknown dtype" + expected


def test_missing_dot_is_actionable_and_does_not_replace_output(tmp_path, monkeypatch):
    output = tmp_path / "existing.svg"
    output.write_text("keep me")
    metadata = output.with_suffix(".json")
    metadata.write_text("keep metadata")
    monkeypatch.setenv("PATH", "")
    with pytest.raises(RenderError, match="Install Graphviz"):
        render_svg(sample_graph(), output)
    assert output.read_text() == "keep me"
    assert metadata.read_text() == "keep metadata"
    assert set(tmp_path.iterdir()) == {output, metadata}


def test_invalid_output_suffix_and_unwritable_parent(tmp_path):
    with pytest.raises(RenderError, match=".svg"):
        render_svg(sample_graph(), tmp_path / "bad.png")
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    with pytest.raises(RenderError, match="Cannot write SVG"):
        render_svg(sample_graph(), blocker / "graph.svg")


def test_empty_and_isolated_graphs_render(tmp_path):
    for index, graph in enumerate([Graph(), Graph([Node(9, "Unknown", "isolated")])]):
        ET.parse(render_svg(graph, tmp_path / f"{index}.svg"))


def test_graphviz_placeholders_are_literal_in_labels_and_tooltips(tmp_path):
    name = r"layer\N\G\E\T\H\L\\end"
    graph = Graph([Node(0, "Custom", name, attributes={"pattern": name})])
    root = ET.parse(render_svg(graph, tmp_path / "literal.svg", label_detail="full")).getroot()
    texts = [element.text for element in root.iter() if element.tag.endswith("}text")]
    assert name in texts
    assert "pattern:" + name in "".join(texts)
    assert name in " ".join(value for element in root.iter() for value in element.attrib.values())


def test_invalid_xml_characters_are_sanitized_everywhere(tmp_path):
    graph = Graph([Node(0, "Custom\ufffe", "a\x00b\ud800", attributes={"key": "\x01"})])
    root = ET.parse(render_svg(graph, tmp_path / "sanitized.svg", label_detail="full")).getroot()
    assert "a�b�" in " ".join(root.itertext())


def test_timeout_preserves_previous_output_and_cleans_temporary_files(tmp_path, monkeypatch):
    def expire(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    output = tmp_path / "existing.svg"
    output.write_text("keep me")
    metadata = output.with_suffix(".json")
    metadata.write_text("keep metadata")
    monkeypatch.setattr(subprocess, "run", expire)
    with pytest.raises(RenderError, match="exceeded the 1-second"):
        render_svg(sample_graph(), output, timeout=1)
    assert output.read_text() == "keep me"
    assert metadata.read_text() == "keep metadata"
    assert set(tmp_path.iterdir()) == {output, metadata}


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_invalid_timeout_rejected(tmp_path, timeout):
    with pytest.raises(RenderError, match="positive finite"):
        render_svg(sample_graph(), tmp_path / "bad.svg", timeout=timeout)


def test_large_graph_retains_nodes_and_uses_bounded_layout():
    graph = Graph(Node(index, "Relu") for index in range(1001))
    dot = to_dot(graph)
    assert dot.graph_attr["splines"] == "line"
    assert dot.graph_attr["nslimit"] == "2"
    assert sum('id="node-' in line for line in dot.body) == 1001


def test_large_graph_uses_polyline_only_for_actual_references():
    tensor = TensorMetadata()
    source = Node(0, "Split", output_ports={0: tensor, 1: tensor})
    source.visual.reference_outputs.add(1)
    graph = Graph(
        [
            source,
            Node(1, "Target", input_ports={0: tensor}),
            *(Node(index, "Relu") for index in range(2, 1001)),
        ],
        [Edge(0, 0, 1, 0)],
    )
    direct = to_dot(graph)
    assert direct.graph_attr["splines"] == "line"
    assert "n0:out_0:s -> n1:in_0:n" in direct.source
    source.visual.reference_outputs.add(0)
    referenced = to_dot(graph)
    assert referenced.graph_attr["splines"] == "polyline"
    assert "n0:out_0:s -> n1:in_0:n" not in referenced.source
    assert "ref-0-0-1-0" in referenced.source
    assert sum('id="node-' in line for line in referenced.body) == 1001


def test_compact_labels_keep_short_scalar_attributes_and_export_all_metadata(tmp_path):
    node = Node(
        7,
        "MatMul",
        "a very long node name " * 12,
        attributes={
            "transpose_a": False,
            "transpose_b": True,
            "long": "x" * 80,
            "axes": [1, 2, 3],
        },
    )
    output = render_svg(Graph([node]), tmp_path / "compact.svg")
    ns = {"s": "http://www.w3.org/2000/svg"}
    root = ET.parse(output).getroot()
    text = " ".join(element.text or "" for element in root.findall(".//s:text", ns))
    assert node.name not in text
    assert "transpose_a:False" in text and "transpose_b:True" in text
    assert "axes" not in text and "x" * 80 not in text
    assert node.name in " ".join(
        value for element in root.iter() for value in element.attrib.values()
    )
    metadata = json.loads(output.with_suffix(".json").read_text())
    assert metadata["nodes"]["7"]["name"] == node.name
    assert metadata["nodes"]["7"]["attributes"] == node.attributes
    assert metadata["rendering"] == {
        "label_detail": "compact",
        "rendered_edges": 0,
        "referenced_edges": 0,
    }


def test_references_preserve_every_endpoint_and_other_output_edges(tmp_path):
    graph = Graph(
        [
            Node(-7, "Split", output_ports={0: TENSOR, 2: TENSOR}),
            Node(-9, "Target", input_ports={1: TENSOR, 3: TENSOR, 4: TENSOR}),
        ],
        [Edge(-7, 2, -9, 1, TENSOR), Edge(-7, 0, -9, 3, TENSOR), Edge(-7, 2, -9, 4, TENSOR)],
    )
    graph.nodes[-7].visual.reference_outputs.add(2)
    before = snapshot_graph(graph)
    output = render_svg(graph, tmp_path / "references.svg")
    assert snapshot_graph(graph) == before
    ns = {"s": "http://www.w3.org/2000/svg"}
    root = ET.parse(output).getroot()
    text = " ".join(root.itertext())
    assert "in 1 ← @-7:2" in text and "in 4 ← @-7:2" in text
    assert "out 2 → @-7:2 (2 uses)" in text
    assert "References:" in text
    assert len(root.findall(".//s:g[@class='edge']", ns)) == 1
    for port in (1, 4):
        marker = root.find(f".//s:g[@id='a_ref--7-2--9-{port}']", ns)
        assert marker is not None
        link = marker.find("s:a", ns)
        assert link.get("{http://www.w3.org/1999/xlink}href") == "#node--7"
        assert link.get("{http://www.w3.org/1999/xlink}title") == f"-7:out 2 → -9:in {port}"
    data = json.loads(output.with_suffix(".json").read_text())
    assert len(data["view"]["edges"]) == 3
    assert [edge["representation"] for edge in data["view"]["edges"]] == [
        "reference",
        "edge",
        "reference",
    ]
    assert data["rendering"]["rendered_edges"] == 1
    assert data["rendering"]["referenced_edges"] == 2


def test_legend_is_absent_without_references():
    assert "References:" not in to_dot(sample_graph()).source


@pytest.mark.parametrize("label_detail", ["unknown", "", None])
def test_invalid_label_detail_fails_before_creating_output(tmp_path, label_detail):
    with pytest.raises(ValueError, match="Label detail"):
        render_svg(sample_graph(), tmp_path / "invalid.svg", label_detail=label_detail)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("kind", ["same", "symlink", "hardlink"])
def test_standalone_renderer_rejects_svg_metadata_aliases(tmp_path, kind):
    output = tmp_path / "out.svg"
    output.write_text("previous SVG")
    metadata = tmp_path / "metadata.json"
    if kind == "same":
        metadata = output
    elif kind == "symlink":
        metadata.symlink_to(output)
    else:
        os.link(output, metadata)
    with pytest.raises(RenderError, match="different"):
        render_svg(sample_graph(), output, metadata=metadata)
    assert output.read_text() == "previous SVG"


@pytest.mark.parametrize("existing_metadata", [False, True])
def test_pair_publication_rolls_back_metadata_when_svg_replace_fails(
    tmp_path, monkeypatch, existing_metadata
):
    output = tmp_path / "existing.svg"
    output.write_text("previous SVG")
    metadata = tmp_path / "elsewhere" / "metadata.json"
    metadata.parent.mkdir()
    if existing_metadata:
        metadata.write_text("previous metadata")
    original_replace = os.replace

    def fail_svg(source, destination):
        if Path(destination) == output:
            raise OSError("simulated SVG replacement failure")
        return original_replace(source, destination)

    monkeypatch.setattr("model_graph_viewer._paths.os.replace", fail_svg)
    with pytest.raises(RenderError, match="simulated SVG replacement failure"):
        render_svg(sample_graph(), output, metadata=metadata)
    assert output.read_text() == "previous SVG"
    if existing_metadata:
        assert metadata.read_text() == "previous metadata"
    else:
        assert not metadata.exists()
    assert not list(tmp_path.rglob(".model-graph-*"))


def test_metadata_serialization_failure_keeps_existing_pair(tmp_path, monkeypatch):
    output = tmp_path / "existing.svg"
    metadata = output.with_suffix(".json")
    output.write_text("previous SVG")
    metadata.write_text("previous metadata")

    def fail_serialization(*args, **kwargs):
        raise TypeError("simulated unsupported attribute")

    monkeypatch.setattr("model_graph_viewer.render.json.dumps", fail_serialization)
    with pytest.raises(RenderError, match="Cannot serialize metadata.*existing.json"):
        render_svg(sample_graph(), output)
    assert output.read_text() == "previous SVG"
    assert metadata.read_text() == "previous metadata"
    assert set(tmp_path.iterdir()) == {output, metadata}


def test_partial_metadata_write_keeps_existing_pair(tmp_path, monkeypatch):
    output = tmp_path / "existing.svg"
    metadata = output.with_suffix(".json")
    output.write_text("previous SVG")
    metadata.write_text("previous metadata")
    original_write = Path.write_text

    def fail_metadata(path, text, *args, **kwargs):
        if path.name == "metadata.json" and path.parent.name.startswith(".model-graph-metadata-"):
            original_write(path, "partial", *args, **kwargs)
            raise OSError("simulated metadata disk full")
        return original_write(path, text, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_metadata)
    with pytest.raises(RenderError, match="simulated metadata disk full"):
        render_svg(sample_graph(), output)
    assert output.read_text() == "previous SVG"
    assert metadata.read_text() == "previous metadata"
    assert set(tmp_path.iterdir()) == {output, metadata}


def test_pair_rollback_restores_existing_metadata_symlink(tmp_path, monkeypatch):
    output = tmp_path / "graph.svg"
    output.write_text("previous SVG")
    target = tmp_path / "original.json"
    target.write_text("previous metadata")
    metadata = output.with_suffix(".json")
    metadata.symlink_to(target.name)
    original_replace = os.replace

    def fail_svg(source, destination):
        if Path(destination) == output:
            raise OSError("simulated SVG replacement failure")
        return original_replace(source, destination)

    monkeypatch.setattr("model_graph_viewer._paths.os.replace", fail_svg)
    with pytest.raises(RenderError, match="simulated SVG replacement failure"):
        render_svg(sample_graph(), output)
    assert metadata.is_symlink() and metadata.readlink() == Path(target.name)
    assert metadata.read_text() == target.read_text() == "previous metadata"
    assert output.read_text() == "previous SVG"
    assert not list(tmp_path.glob(".model-graph-*"))


def test_graphviz_failure_preserves_existing_pair(tmp_path, monkeypatch):
    output = tmp_path / "graph.svg"
    metadata = output.with_suffix(".json")
    output.write_text("previous SVG")
    metadata.write_text("previous metadata")

    def fail_layout(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0], stderr="simulated invalid DOT")

    monkeypatch.setattr(subprocess, "run", fail_layout)
    with pytest.raises(RenderError, match="simulated invalid DOT"):
        render_svg(sample_graph(), output)
    assert output.read_text() == "previous SVG"
    assert metadata.read_text() == "previous metadata"
    assert set(tmp_path.iterdir()) == {output, metadata}
