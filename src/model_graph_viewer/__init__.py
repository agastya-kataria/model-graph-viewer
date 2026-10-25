"""Framework-independent graph visualization with configurable transformation passes."""

from .api import ModelLoader, VisualizationResult, load_model, visualize
from .graph import ConstantAnnotation, Edge, Graph, Node, TensorMetadata, Visualization
from .passes import (
    HideConstantConverts,
    HideConstants,
    HighlightNodes,
    ReferenceHighFanout,
    apply_passes,
)
from .render import RenderError, render_svg

__all__ = [
    "ConstantAnnotation",
    "Edge",
    "Graph",
    "HideConstantConverts",
    "HideConstants",
    "HighlightNodes",
    "ModelLoader",
    "Node",
    "ReferenceHighFanout",
    "RenderError",
    "TensorMetadata",
    "Visualization",
    "VisualizationResult",
    "apply_passes",
    "load_model",
    "render_svg",
    "visualize",
]
