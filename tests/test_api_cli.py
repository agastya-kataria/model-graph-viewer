import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path

import numpy as np
import openvino as ov
import pytest
from openvino import opset13 as ops

from model_graph_viewer import (
    Graph,
    HideConstantConverts,
    HideConstants,
    HighlightNodes,
    Node,
    load_model,
    visualize,
)
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


@pytest.mark.parametrize("kind", ["weights", "symlink_weights", "hardlink_xml", "hardlink_weights"])
def test_stats_cannot_overwrite_input_or_weight_aliases(
    kind, model_path, tmp_path, monkeypatch, capsys
):
    weights = model_path.with_suffix(".bin")
    before = (model_path.read_bytes(), weights.read_bytes())
    stats = tmp_path / "statistics.json"
    if kind == "weights":
        stats = weights
    elif kind == "symlink_weights":
        stats.symlink_to(weights)
    elif kind == "hardlink_xml":
        os.link(model_path, stats)
    else:
        os.link(weights, stats)

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid output paths must be rejected before loading")

    monkeypatch.setattr("model_graph_viewer.api.load_model", forbidden)
    output = tmp_path / "out.svg"
    with pytest.raises(SystemExit) as caught:
        main(["--model", str(model_path), "--output", str(output), "--stats", str(stats)])
    assert caught.value.code == 2
    assert "must not overwrite" in capsys.readouterr().err
    assert (model_path.read_bytes(), weights.read_bytes()) == before
    assert not output.exists()


@pytest.mark.parametrize("entrypoint", ["api", "cli"])
def test_svg_cannot_overwrite_target_of_symlinked_model(
    entrypoint, model_path, tmp_path, monkeypatch
):
    target = tmp_path / "original.svg"
    before = model_path.read_bytes()
    model_path.rename(target)
    model_path.symlink_to(target)

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid output paths must be rejected before loading")

    monkeypatch.setattr("model_graph_viewer.api.load_model", forbidden)
    if entrypoint == "api":
        with pytest.raises(ValueError, match="must not overwrite"):
            visualize(model_path, target)
    else:
        with pytest.raises(SystemExit) as caught:
            main(["--model", str(model_path), "--output", str(target)])
        assert caught.value.code == 2
    assert target.read_bytes() == model_path.read_bytes() == before


def test_stats_protects_resolved_weights_for_symlinked_xml(model_path, tmp_path):
    weights = model_path.with_suffix(".bin")
    alias = tmp_path / "alias.xml"
    alias.symlink_to(model_path)
    before = (model_path.read_bytes(), weights.read_bytes())
    output = tmp_path / "out.svg"
    with pytest.raises(SystemExit) as caught:
        main(["--model", str(alias), "--output", str(output), "--stats", str(weights)])
    assert caught.value.code == 2
    assert (model_path.read_bytes(), weights.read_bytes()) == before
    assert not output.exists()


@pytest.mark.parametrize("kind", ["same_path", "symlink", "hardlink"])
def test_stats_cannot_alias_svg(kind, model_path, tmp_path):
    output = tmp_path / "out.svg"
    output.write_text("previous SVG")
    stats = tmp_path / "statistics.json"
    if kind == "same_path":
        stats = output
    elif kind == "symlink":
        stats.symlink_to(output)
    else:
        os.link(output, stats)
    before = model_path.read_bytes()
    with pytest.raises(SystemExit) as caught:
        main(["--model", str(model_path), "--output", str(output), "--stats", str(stats)])
    assert caught.value.code == 2
    assert output.read_text() == "previous SVG"
    assert model_path.read_bytes() == before


def test_expanded_model_path_cannot_be_overwritten_by_stats(model_path, tmp_path):
    spelling = "~/" + os.path.relpath(model_path, Path.home())
    before = model_path.read_bytes()
    output = tmp_path / "out.svg"
    with pytest.raises(SystemExit) as caught:
        main(["--model", spelling, "--output", str(output), "--stats", str(model_path)])
    assert caught.value.code == 2
    assert model_path.read_bytes() == before
    assert not output.exists()


def test_stats_symlink_loop_has_actionable_error_without_traceback(model_path, tmp_path, capsys):
    stats = tmp_path / "loop.json"
    stats.symlink_to(stats.name)
    output = tmp_path / "out.svg"
    with pytest.raises(SystemExit) as caught:
        main(["--model", str(model_path), "--output", str(output), "--stats", str(stats)])
    assert caught.value.code == 2
    error = capsys.readouterr().err
    assert "loop.json" in error and "Traceback" not in error
    assert not output.exists()


def test_failed_stats_write_keeps_existing_stats_and_cleans_staging(
    model_path, tmp_path, monkeypatch, capsys
):
    stats = tmp_path / "statistics.txt"
    stats.write_text("previous statistics")
    original_write = Path.write_text

    def fail_statistics(path, text, *args, **kwargs):
        if path.name == "statistics.json" and path.parent.name.startswith(".model-graph-stats-"):
            original_write(path, "partial", *args, **kwargs)
            raise OSError("simulated disk full")
        return original_write(path, text, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_statistics)
    output = tmp_path / "out.svg"
    assert main(["--model", str(model_path), "--output", str(output), "--stats", str(stats)]) == 1
    assert "simulated disk full" in capsys.readouterr().err
    assert stats.read_text() == "previous statistics"
    assert not list(tmp_path.glob(".model-graph-stats-*"))
    ET.parse(output)


def test_successful_api_and_cli_preserve_xml_and_weights_through_all_passes(tmp_path):
    parameter = ops.parameter([2], ov.Type.f32)
    constant = ops.constant(np.array([1.0, 2.0], dtype=np.float16))
    converted = ops.convert(constant, ov.Type.f32)
    total = ops.add(parameter, converted, name="sum")
    path = tmp_path / 'model with spaces & "quotes".xml'
    ov.serialize(ov.Model([total], [parameter]), str(path))
    weights = path.with_suffix(".bin")
    before = (path.read_bytes(), weights.read_bytes())
    api_output = tmp_path / "api.svg"
    visualize(
        path,
        api_output,
        [HideConstantConverts(), HideConstants(), HighlightNodes(["sum"])],
    )
    cli_output = tmp_path / "cli.svg"
    stats = tmp_path / "counts with spaces.txt"
    assert (
        main(
            [
                "--model",
                str(path),
                "--output",
                str(cli_output),
                "--passes",
                "hide-constant-converts,hide-constants,highlight",
                "--highlight",
                "sum",
                "--stats",
                str(stats),
            ]
        )
        == 0
    )
    assert (path.read_bytes(), weights.read_bytes()) == before
    assert json.loads(stats.read_text())["after_nodes"] == 3
    ET.parse(api_output)
    ET.parse(cli_output)
    assert not list(tmp_path.glob(".model-graph-*"))
