import subprocess
import xml.etree.ElementTree as ET

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
    output = render_svg(graph, tmp_path / "nested" / "graph.svg")
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
    assert list(output.parent.iterdir()) == [output]


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
    monkeypatch.setenv("PATH", "")
    with pytest.raises(RenderError, match="Install Graphviz"):
        render_svg(sample_graph(), output)
    assert output.read_text() == "keep me"
    assert list(tmp_path.iterdir()) == [output]


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
    root = ET.parse(render_svg(graph, tmp_path / "literal.svg")).getroot()
    texts = [element.text for element in root.iter() if element.tag.endswith("}text")]
    assert name in texts
    assert "pattern:" + name in "".join(texts)
    assert name in " ".join(value for element in root.iter() for value in element.attrib.values())


def test_invalid_xml_characters_are_sanitized_everywhere(tmp_path):
    graph = Graph([Node(0, "Custom\ufffe", "a\x00b\ud800", attributes={"key": "\x01"})])
    root = ET.parse(render_svg(graph, tmp_path / "sanitized.svg")).getroot()
    assert "a�b�" in " ".join(root.itertext())


def test_timeout_preserves_previous_output_and_cleans_temporary_files(tmp_path, monkeypatch):
    def expire(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    output = tmp_path / "existing.svg"
    output.write_text("keep me")
    monkeypatch.setattr(subprocess, "run", expire)
    with pytest.raises(RenderError, match="exceeded the 1-second"):
        render_svg(sample_graph(), output, timeout=1)
    assert output.read_text() == "keep me"
    assert list(tmp_path.iterdir()) == [output]


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
