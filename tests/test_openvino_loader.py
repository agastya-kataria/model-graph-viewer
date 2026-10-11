"""Adapter tests use real OpenVINO models; no network or inference is needed."""

import json
from dataclasses import asdict

import numpy as np
import openvino as ov
import pytest
from openvino import opset13 as ops

from model_graph_viewer.openvino_loader import ModelLoadError, OpenVINOLoader


@pytest.fixture
def loader():
    return OpenVINOLoader()


def save_model(model, path):
    ov.serialize(model, str(path))
    return path


def test_split_ports_duplicate_names_and_parallel_edges(loader, tmp_path):
    parameter = ops.parameter([-1, 4], ov.Type.f32, name="duplicate")
    split = ops.split(parameter, ops.constant(1), 2, name="duplicate")
    add = ops.add(split.output(0), split.output(1), name="sum")
    model = ov.Model([add], [parameter])
    graph = loader.load(save_model(model, tmp_path / "split.xml"))

    split_node = next(n for n in graph.nodes.values() if n.node_type == "Split")
    add_node = next(n for n in graph.nodes.values() if n.node_type == "Add")
    duplicates = [n for n in graph.nodes.values() if n.name == "duplicate"]
    assert len(duplicates) == 2
    assert len({n.node_id for n in duplicates}) == 2
    assert set(split_node.input_ports) == {0, 1}
    assert set(split_node.output_ports) == {0, 1}
    assert split_node.output_ports[0].shape == ("?", 2)
    assert split_node.output_ports[1].dtype == "f32"
    edges = [e for e in graph.edges if e.from_node == split_node.node_id]
    assert {(e.from_port, e.to_node, e.to_port) for e in edges} == {
        (0, add_node.node_id, 0),
        (1, add_node.node_id, 1),
    }
    # Serialized Split output IDs follow input IDs; runtime outputs start at 0.
    assert split_node.attributes == {"num_splits": 2}
    graph.validate()


@pytest.mark.parametrize(
    ("partial_shape", "expected"),
    [
        (ov.PartialShape.dynamic(), "?"),
        (ov.PartialShape([]), ()),
        (ov.PartialShape([-1, 7]), ("?", 7)),
        (
            ov.PartialShape([ov.Dimension(1, 8), ov.Dimension(0, 4), ov.Dimension(2, -1)]),
            ("1..8", "0..4", "2..?"),
        ),
    ],
)
def test_partial_shapes_are_not_confused_with_scalars(loader, partial_shape, expected):
    parameter = ops.parameter(partial_shape, ov.Type.f32, name="input")
    graph = loader.from_model(ov.Model([parameter], [parameter]))
    node = next(n for n in graph.nodes.values() if n.node_type == "Parameter")
    assert node.output_ports[0].shape == expected
    assert all(edge.tensor.shape == expected for edge in graph.edges)


def test_constant_data_and_attributes_are_never_read(loader, monkeypatch):
    parameter = ops.parameter([2], ov.Type.f32)
    constant = ops.constant(np.array([913.25, -71.5], dtype=np.float32))
    model = ov.Model([ops.add(parameter, constant)], [parameter])

    def forbidden(*args, **kwargs):
        raise AssertionError("The adapter must never read Constant tensor values")

    for method in ("get_attributes", "get_data", "get_vector", "get_tensor_view"):
        if hasattr(type(constant), method):
            monkeypatch.setattr(type(constant), method, forbidden)
    graph = loader.from_model(model)
    node = next(n for n in graph.nodes.values() if n.node_type == "Constant")
    assert node.attributes == {}
    assert node.output_ports[0].shape == (2,)
    assert node.output_ports[0].dtype == "f32"
    serialized = json.dumps([asdict(n) for n in graph.nodes.values()], allow_nan=False)
    assert "913.25" not in serialized
    assert "-71.5" not in serialized


def test_attributes_are_portable_and_model_is_unmodified(loader):
    parameter = ops.parameter([1, 2, 6, 6], ov.Type.f32, name="input")
    weights = ops.constant(np.ones((3, 2, 3, 3), dtype=np.float32))
    convolution = ops.convolution(parameter, weights, [1, 1], [0, 0], [0, 0], [1, 1])
    model = ov.Model([convolution], [parameter])
    before = [
        (n.get_instance_id(), n.get_friendly_name(), n.get_type_name())
        for n in model.get_ordered_ops()
    ]
    graph = loader.from_model(model)
    node = next(n for n in graph.nodes.values() if n.node_type == "Convolution")
    assert node.attributes["strides"] == [1, 1]
    json.dumps([asdict(n) for n in graph.nodes.values()], allow_nan=False)
    node.name = "changed in detached graph"
    node.attributes["strides"][0] = 99
    assert convolution.get_attributes()["strides"] == [1, 1]
    assert before == [
        (n.get_instance_id(), n.get_friendly_name(), n.get_type_name())
        for n in model.get_ordered_ops()
    ]


def test_stateful_sink_is_kept(loader, tmp_path):
    parameter = ops.parameter([2], ov.Type.f32, name="initial")
    read = ops.read_value(parameter, "memory")
    updated = ops.add(read, parameter)
    assign = ops.assign(updated, "memory")
    model = ov.Model([ops.result(read)], [assign], [parameter])
    graph = loader.load(save_model(model, tmp_path / "state.xml"))
    sink = next(n for n in graph.nodes.values() if n.node_type == "Assign")
    add = next(n for n in graph.nodes.values() if n.node_type == "Add")
    assert sink.attributes["variable_id"] == "memory"
    assert any(e.from_node == add.node_id and e.to_node == sink.node_id for e in graph.edges)
    graph.validate()


def test_disconnected_parameter_is_kept(loader):
    parameter = ops.parameter([2], ov.Type.f32, name="used")
    unused = ops.parameter([2], ov.Type.f32, name="unused")
    graph = loader.from_model(ov.Model([ops.relu(parameter)], [parameter, unused]))
    isolated = next(n for n in graph.nodes.values() if n.name == "unused")
    assert isolated.node_type == "Parameter"
    assert not graph.incoming(isolated.node_id)
    assert not graph.outgoing(isolated.node_id)
    assert isolated.output_ports[0].shape == (2,)


def test_ids_are_deterministic_across_loading(loader, tmp_path):
    parameter = ops.parameter([2], ov.Type.f32)
    path = save_model(ov.Model([ops.relu(parameter)], [parameter]), tmp_path / "model.xml")
    first = loader.load(path)
    second = loader.load(path)
    assert [asdict(n) for n in first.nodes.values()] == [asdict(n) for n in second.nodes.values()]
    assert list(first.edges) == list(second.edges)


def test_weightless_ir_does_not_require_bin_or_compilation(loader, tmp_path, monkeypatch):
    parameter = ops.parameter([2], ov.Type.f32)
    path = save_model(ov.Model([ops.relu(parameter)], [parameter]), tmp_path / "model.xml")
    path.with_suffix(".bin").unlink()

    def forbidden(*args, **kwargs):
        raise AssertionError("Loading the visualization must not compile a model")

    monkeypatch.setattr(ov.Core, "compile_model", forbidden)
    assert len(loader.load(path).nodes) == 3


def test_missing_required_weights_is_actionable(loader, tmp_path):
    parameter = ops.parameter([2], ov.Type.f32)
    constant = ops.constant(np.array([1.0, 2.0], dtype=np.float32))
    path = save_model(
        ov.Model([ops.add(parameter, constant)], [parameter]), tmp_path / "weighted.xml"
    )
    path.with_suffix(".bin").unlink()
    with pytest.raises(ModelLoadError, match=r"required weights.*matching \.bin"):
        loader.load(path)


def test_missing_invalid_and_unsupported_models(loader, tmp_path):
    with pytest.raises(ModelLoadError, match="does not exist"):
        loader.load(tmp_path / "missing.xml")
    with pytest.raises(ModelLoadError, match=r"Expected an OpenVINO \.xml"):
        loader.load(tmp_path / "model.onnx")
    directory = tmp_path / "directory.xml"
    directory.mkdir()
    with pytest.raises(ModelLoadError, match="does not exist"):
        loader.load(directory)
    invalid = tmp_path / "invalid.xml"
    invalid.write_text("<not-a-model>")
    with pytest.raises(ModelLoadError, match="XML is valid"):
        loader.load(invalid)
