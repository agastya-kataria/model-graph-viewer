import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict

import numpy as np
import openvino as ov
import pytest
from openvino import opset13 as ops

from model_graph_viewer import Graph, HideConstants, HighlightNodes, Node, load_model, visualize
from model_graph_viewer.cli import main


@pytest.fixture
def model_path(tmp_path):
    parameter = ops.parameter([1, 4], ov.Type.f32, name="input")
    constant = ops.constant(np.ones((1, 4), dtype=np.float32), name="bias")
    add = ops.add(parameter, constant, name="sum")
    model = ov.Model([add], [parameter], "small")
    path = tmp_path / "model.xml"
    ov.serialize(model, path, path.with_suffix(".bin"))
    return path


def test_api_end_to_end(model_path, tmp_path):
    result = visualize(
        model_path, tmp_path / "result.svg", [HideConstants(), HighlightNodes(["sum"])]
    )
    assert (result.before_nodes, result.before_edges) == (4, 3)
    assert (result.after_nodes, result.after_edges) == (3, 2)
    assert set(result.timings) == {"load_construct_seconds", "transform_seconds", "render_seconds"}
    assert all(value >= 0 for value in result.timings.values())
    ET.parse(result.output_path)
    assert json.loads(json.dumps(asdict(result)))["after_nodes"] == 3


def test_custom_loader_needs_no_changes_to_pipeline_or_renderer(tmp_path):
    class Loader:
        def load(self, path):
            assert str(path) == "anything.custom"
            return Graph([Node(42, "Custom", "custom")])

    assert load_model("anything.custom", loader=Loader()).nodes[42].name == "custom"
    result = visualize("anything.custom", tmp_path / "custom.svg", loader=Loader())
    assert result.after_nodes == 1


def test_module_cli_end_to_end_outside_project(model_path, tmp_path):
    output = tmp_path / "cli.svg"
    stats = tmp_path / "stats.json"
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "model_graph_viewer",
            "--model",
            str(model_path),
            "--output",
            str(output),
            "--passes",
            "hide-constants,highlight",
            "--highlight",
            "sum",
            "--stats",
            str(stats),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    assert "4 → 3 nodes" in process.stdout
    assert json.loads(stats.read_text())["after_edges"] == 2
    root = ET.parse(output).getroot()
    assert "[1]: f32[1, 4]" in " ".join(root.itertext())


def test_default_cli_applies_no_passes(model_path, tmp_path):
    stats = tmp_path / "stats.json"
    assert (
        main(
            [
                "--model",
                str(model_path),
                "--output",
                str(tmp_path / "plain.svg"),
                "--stats",
                str(stats),
            ]
        )
        == 0
    )
    counts = json.loads(stats.read_text())
    assert counts["before_nodes"] == counts["after_nodes"] == 4


@pytest.mark.parametrize(
    "extra",
    [["--passes", "highlight"], ["--highlight", "sum"], ["--passes", "hide-constants,"]],
)
def test_invalid_cli_configuration_fails_before_loading(extra, tmp_path, capsys):
    with pytest.raises(SystemExit) as caught:
        main(["--model", "absent.xml", "--output", str(tmp_path / "out.svg"), *extra])
    assert caught.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_unknown_pass_and_bad_input_report_no_traceback(tmp_path, capsys):
    base = ["--model", "absent.xml", "--output", str(tmp_path / "out.svg")]
    assert main([*base, "--passes", "typo"]) == 1
    assert "Unknown pass" in capsys.readouterr().err
    assert main(base) == 1
    error = capsys.readouterr().err
    assert "absent.xml" in error and "Traceback" not in error


def test_stats_cannot_overwrite_model_or_svg(model_path, tmp_path):
    before = model_path.read_bytes()
    with pytest.raises(SystemExit):
        main(
            [
                "--model",
                str(model_path),
                "--output",
                str(tmp_path / "out.svg"),
                "--stats",
                str(model_path),
            ]
        )
    assert model_path.read_bytes() == before
