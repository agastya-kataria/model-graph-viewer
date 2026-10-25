"""The artifact checker must reject incomplete or misleading visualizations."""

import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from model_graph_viewer import Edge, Graph, Node, ReferenceHighFanout, TensorMetadata, render_svg
from model_graph_viewer.metadata import snapshot_graph


@pytest.fixture(scope="module")
def benchmark():
    path = Path(__file__).resolve().parents[1] / "scripts" / "benchmark.py"
    spec = importlib.util.spec_from_file_location("benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def artifact(tmp_path):
    tensor = TensorMetadata(("?", 2), "f32")
    graph = Graph(
        [
            Node(-8, "Split", "duplicate", output_ports={0: tensor, 1: tensor}),
            Node(
                4, "Add", "duplicate", input_ports={0: tensor, 1: tensor}, output_ports={0: tensor}
            ),
            Node(5, "Add", input_ports={0: tensor, 1: tensor}),
            Node(6, "Relu", input_ports={0: tensor}),
        ],
        [
            Edge(-8, 0, 4, 0, tensor),
            Edge(-8, 1, 4, 1, tensor),
            Edge(-8, 0, 5, 0, tensor),
            Edge(-8, 1, 5, 1, tensor),
            Edge(4, 0, 6, 0, tensor),
        ],
    )
    original = snapshot_graph(graph)
    ReferenceHighFanout().apply(graph)
    svg = render_svg(graph, tmp_path / "view.svg", _original=original)
    return graph, svg, original


def test_checker_recovers_multioutput_repeated_connections_and_negative_ids(benchmark, artifact):
    graph, svg, original = artifact
    result = benchmark.verify_svg(graph, svg, original, label_detail="compact")
    assert result["nodes"] == 4
    assert result["logical_edges"] == 5
    assert result["rendered_edges"] == 1
    assert result["referenced_edges"] == 4
    assert result["reference_sources"] == 1
    assert result["reference_outputs"] == 2


@pytest.mark.parametrize("corruption", ["reference_link", "direct_endpoint", "source_count"])
def test_checker_rejects_corrupted_svg(benchmark, artifact, corruption):
    graph, svg, original = artifact
    tree = ET.parse(svg)
    root = tree.getroot()
    if corruption == "reference_link":
        reference = next(e for e in root.iter() if benchmark.REFERENCE.fullmatch(e.get("id", "")))
        link = next(reference.iter(f"{benchmark.SVG}a"))
        link.set(f"{benchmark.XLINK}href", "#node-999")
    elif corruption == "direct_endpoint":
        link = next(
            e for e in root.iter() if e.get(f"{benchmark.XLINK}title") == "4:out 0 → 6:in 0"
        )
        link.set(f"{benchmark.XLINK}title", "4:out 0 → 6:in 1")
    else:
        text = next(e for e in root.iter(f"{benchmark.SVG}text") if "(2 uses)" in (e.text or ""))
        text.text = text.text.replace("(2 uses)", "(3 uses)")
    tree.write(svg, encoding="utf-8", xml_declaration=True)
    with pytest.raises(RuntimeError):
        benchmark.verify_svg(graph, svg, original, label_detail="compact")


@pytest.mark.parametrize("corruption", ["missing_connection", "tensor", "original_name"])
def test_checker_rejects_corrupted_metadata(benchmark, artifact, corruption):
    graph, svg, original = artifact
    path = svg.with_suffix(".json")
    metadata = json.loads(path.read_text())
    if corruption == "missing_connection":
        metadata["view"]["edges"].pop()
    elif corruption == "tensor":
        metadata["view"]["edges"][0]["tensor"]["shape"] = [999]
    else:
        metadata["nodes"]["-8"]["name"] = "changed"
    path.write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError):
        benchmark.verify_svg(graph, svg, original, label_detail="compact")
