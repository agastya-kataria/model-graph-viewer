"""Render a framework-independent graph using Graphviz's hierarchical layout."""

from __future__ import annotations

import json
import math
import subprocess
import textwrap
from collections import Counter
from contextlib import ExitStack
from html import escape
from pathlib import Path
from tempfile import TemporaryDirectory

import graphviz

from ._paths import _expand, _same_file, publish_files
from .graph import ConstantAnnotation, Edge, Graph, Node, TensorMetadata
from .metadata import metadata_path_for, visualization_metadata

# Retain the starter example's palette, with a neutral fallback for new operation types.
NODE_FILLCOLOR_BY_TYPE = {
    "Input": "#EDEADE",
    "Parameter": "#EDEADE",
    "Constant": "#FFD54F",
    "MatMul": "#90CAF9",
    "Add": "#B39DDB",
    "Output": "#EDEADE",
    "Result": "#EDEADE",
}


class RenderError(RuntimeError):
    """The requested SVG could not be rendered or written."""


def format_tensor(tensor: TensorMetadata) -> str:
    """Keep unknown metadata, dynamic rank, and scalar shapes visibly distinct."""
    dtype = tensor.dtype or "unknown dtype"
    if tensor.shape is None:
        shape = "[unknown shape]"
    elif tensor.shape == "?":
        shape = "[dynamic rank]"
    else:
        shape = "[" + ", ".join(map(str, tensor.shape)) + "]"
    return f"{dtype}{shape}"


def format_constant(annotation: ConstantAnnotation, *, compact: bool = False) -> str:
    label = format_tensor(annotation.tensor)
    if annotation.convert_id is not None:
        if compact:
            label += f" (converted #{annotation.convert_id})"
        else:
            source = f"source: {annotation.source_id}"
            if annotation.source_shared:
                source += ", shared source"
            label += (
                f" (converted {annotation.stored_dtype or '?'}"
                f" → {annotation.tensor.dtype or '?'}; {source}; via: {annotation.convert_id})"
            )
    if annotation.shared:
        identity = (
            annotation.convert_id if annotation.convert_id is not None else annotation.source_id
        )
        label += f" (shared: {identity})"
    return label


def _html_text(value: object, width: int = 64) -> str:
    """Escape label text and wrap long lines without dropping information."""
    if isinstance(value, (list, tuple, dict)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    lines = _xml_text(str(value)).splitlines() or [""]
    chunks = [chunk for line in lines for chunk in (textwrap.wrap(line, width) or [""])]
    # Graphviz expands backslash placeholders even within HTML-like labels.
    return '<BR ALIGN="LEFT"/>'.join(
        escape(chunk.replace("\\", "\\\\"), quote=True) for chunk in chunks
    )


def _xml_text(value: str) -> str:
    """Replace characters forbidden by XML 1.0, including lone surrogates."""
    return "".join(
        char
        if char in "\t\n\r"
        or 0x20 <= ord(char) <= 0xD7FF
        or 0xE000 <= ord(char) <= 0xFFFD
        or 0x10000 <= ord(char) <= 0x10FFFF
        else "�"
        for char in value
    )


def _ports(
    ports: dict[int, TensorMetadata],
    direction: str,
    node_id: int,
    references: dict[int, Edge],
    use_counts: dict[int, int],
) -> str:
    if not ports:
        return ""
    cells = []
    for index in sorted(ports):
        attributes = f'PORT="{direction}_{index}" BORDER="1" COLOR="#D1D5DB"'
        label = f"{direction} {index}"
        if direction == "in" and index in references:
            edge = references[index]
            label += f" ← @{edge.from_node}:{edge.from_port}"
            attributes += (
                f' ID="ref-{edge.from_node}-{edge.from_port}-{node_id}-{index}"'
                f' HREF="#node-{edge.from_node}"'
                f' TOOLTIP="{edge.from_node}:out {edge.from_port} → {node_id}:in {index}"'
            )
        elif direction == "out" and index in use_counts:
            label += f" → @{node_id}:{index} ({use_counts[index]} uses)"
            attributes += f' ID="source-{node_id}-{index}" HREF="#node-{node_id}"'
        cells.append(f'<TD {attributes}><FONT POINT-SIZE="9">{_html_text(label)}</FONT></TD>')
    return (
        '<TR><TD><TABLE BORDER="0" CELLBORDER="0" CELLSPACING="2">'
        f"<TR>{''.join(cells)}</TR></TABLE></TD></TR>"
    )


def _validate_label_detail(label_detail: str) -> None:
    if label_detail not in {"compact", "full"}:
        raise ValueError("Label detail must be 'compact' or 'full'.")


def node_label(
    node: Node,
    *,
    label_detail: str = "compact",
    references: dict[int, Edge] | None = None,
    use_counts: dict[int, int] | None = None,
) -> str:
    _validate_label_detail(label_detail)
    references = references or {}
    use_counts = use_counts or {}
    border = escape(node.visual.border_color or "#64748B", quote=True)
    fill = NODE_FILLCOLOR_BY_TYPE.get(node.node_type, "#F1F5F9")
    rows = [_ports(node.input_ports, "in", node.node_id, references, {})]
    rows.append(f'<TR><TD BGCOLOR="{fill}"><B>{_html_text(node.node_type)}</B></TD></TR>')
    if node.name and label_detail == "full":
        rows.append(f'<TR><TD ALIGN="LEFT">{_html_text(node.name)}</TD></TR>')
    rows.append(f'<TR><TD ALIGN="LEFT"><FONT COLOR="#64748B">id: {node.node_id}</FONT></TD></TR>')
    for key, value in sorted(node.attributes.items()):
        if label_detail == "compact":
            if not isinstance(value, (str, int, float, bool, type(None))):
                continue
            if len(f"{key}: {value}") > 40:
                continue
        rows.append(f'<TR><TD ALIGN="LEFT">{_html_text(f"{key}: ")}{_html_text(value)}</TD></TR>')
    for port, annotation in sorted(node.visual.hidden_inputs.items()):
        label = f"[{port}]: {format_constant(annotation, compact=label_detail == 'compact')}"
        rows.append(f'<TR><TD ALIGN="LEFT" BGCOLOR="#FFF8E1">{_html_text(label)}</TD></TR>')
    rows.append(_ports(node.output_ports, "out", node.node_id, {}, use_counts))
    border_width = "2" if node.visual.border_color else "1"
    return (
        f'<<TABLE BORDER="{border_width}" COLOR="{border}" CELLBORDER="0" '
        'CELLSPACING="0" CELLPADDING="4">' + "".join(rows) + "</TABLE>>"
    )


def to_dot(graph: Graph, *, label_detail: str = "compact") -> graphviz.Digraph:
    """Build DOT without invoking Graphviz, useful for testing and custom exporters."""
    graph.validate()
    _validate_label_detail(label_detail)
    drawing = graphviz.Digraph(format="svg", engine="dot", strict=False)
    drawing.graph_attr.update(
        rankdir="TB", ranksep="0.45", nodesep="0.25", bgcolor="white", splines="polyline"
    )
    if len(graph.nodes) > 1000:
        # Long skip connections create many virtual layout nodes in dot.
        # Bound rank optimization; routing is selected after counting references.
        drawing.graph_attr.update(nslimit="2", nslimit1="2", mclimit="0.1")
    drawing.node_attr.update(shape="plain", fontname="Helvetica", fontsize="10")
    drawing.edge_attr.update(arrowsize="0.6", fontsize="9", fontname="Helvetica", color="#64748B")
    references: dict[int, dict[int, Edge]] = {}
    uses: dict[int, Counter] = {}
    for edge in graph.edges:
        if edge.from_port in graph.nodes[edge.from_node].visual.reference_outputs:
            references.setdefault(edge.to_node, {})[edge.to_port] = edge
            uses.setdefault(edge.from_node, Counter())[edge.from_port] += 1
    if len(graph.nodes) > 1000 and not references:
        # Obstacle routing of unreduced global fan-out is prohibitively costly.
        # References remove that pressure and allow polylines around other nodes.
        drawing.graph_attr["splines"] = "line"
    if references:
        drawing.graph_attr.update(
            label=(
                "References: out p → @node:p (N uses); in q ← @node:p links to its source.\\n"
                "Every reference is a connection; complete attributes and connectivity are in JSON."
            ),
            labelloc="t",
            fontname="Helvetica",
            fontsize="10",
        )
    for node in graph.nodes.values():
        drawing.node(
            f"n{node.node_id}",
            label=node_label(
                node,
                label_detail=label_detail,
                references=references.get(node.node_id),
                use_counts=uses.get(node.node_id),
            ),
            id=f"node-{node.node_id}",
            # Tooltip strings pass through DOT decoding and a second Graphviz
            # substitution step; preserve literal backslashes through both.
            tooltip=graphviz.escape(
                graphviz.escape(_xml_text(f"{node.name} ({node.node_type}, id: {node.node_id})"))
            ),
        )
    for index, edge in enumerate(graph.edges):
        if edge.from_port in graph.nodes[edge.from_node].visual.reference_outputs:
            continue
        drawing.edge(
            f"n{edge.from_node}:out_{edge.from_port}:s",
            f"n{edge.to_node}:in_{edge.to_port}:n",
            label=graphviz.escape(_xml_text(format_tensor(edge.tensor))),
            id=f"edge-{index}",
            tooltip=graphviz.escape(
                f"{edge.from_node}:out {edge.from_port} → {edge.to_node}:in {edge.to_port}"
            ),
        )
    return drawing


def render_svg(
    graph: Graph,
    output: str | Path,
    *,
    timeout: float = 300,
    label_detail: str = "compact",
    metadata: str | Path | None = None,
    _original: dict | None = None,
) -> Path:
    """Write an SVG and JSON sidecar, preserving an existing pair on failure."""
    try:
        output = _expand(output)
        metadata_path = metadata_path_for(output, metadata)
        aliases = _same_file(output, metadata_path)
    except ValueError as exc:
        raise RenderError(f"Cannot write SVG to '{output}': {exc}") from exc
    if output.suffix.lower() != ".svg":
        raise RenderError("The output path must end in .svg.")
    if not math.isfinite(timeout) or timeout <= 0:
        raise RenderError("The rendering timeout must be a positive finite number of seconds.")
    if aliases:
        raise RenderError("Metadata output must be different from the SVG output path.")
    drawing = to_dot(graph, label_detail=label_detail)
    try:
        metadata_text = (
            json.dumps(
                visualization_metadata(graph, label_detail, _original),
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        )
    except (TypeError, ValueError, UnicodeError) as exc:
        raise RenderError(f"Cannot serialize metadata to '{metadata_path}': {exc}") from exc
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with ExitStack() as stack:
            directory = stack.enter_context(
                TemporaryDirectory(prefix=".model-graph-", dir=output.parent)
            )
            metadata_directory = stack.enter_context(
                TemporaryDirectory(prefix=".model-graph-metadata-", dir=metadata_path.parent)
            )
            dot_path = Path(directory) / "graph.dot"
            rendered = Path(directory) / "graph.svg"
            staged_metadata = Path(metadata_directory) / "metadata.json"
            dot_path.write_text(drawing.source, encoding="utf-8")
            staged_metadata.write_text(metadata_text, encoding="utf-8")
            subprocess.run(
                ["dot", "-Tsvg", str(dot_path), "-o", str(rendered)],
                capture_output=True,
                check=True,
                timeout=timeout,
            )
            publish_files([(staged_metadata, metadata_path), (rendered, output)])
    except FileNotFoundError as exc:
        if exc.filename != "dot":
            raise RenderError(f"Cannot write SVG to '{output}': {exc}") from exc
        raise RenderError(
            "Graphviz 'dot' was not found. Install Graphviz (brew install graphviz on macOS; "
            "apt install graphviz on Debian/Ubuntu) and ensure dot is on PATH."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RenderError(
            f"Graphviz exceeded the {timeout:g}-second rendering timeout. "
            "Try hiding constants or increase --render-timeout (API: timeout)."
        ) from exc
    except subprocess.CalledProcessError as exc:
        details = (
            exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr
        )
        raise RenderError(f"Graphviz failed to render the graph: {details or exc}") from exc
    except OSError as exc:
        raise RenderError(f"Cannot write SVG to '{output}': {exc}") from exc
    except UnicodeError as exc:
        raise RenderError(
            f"Cannot encode SVG/metadata for '{output}' and '{metadata_path}': {exc}"
        ) from exc
    return output
