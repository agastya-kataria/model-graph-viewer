# Model Graph Viewer

A Python library and CLI for visualizing OpenVINO computational graphs as SVGs. It preserves
operation identity, tensor metadata, and exact input/output port connections while applying ordered
visualization passes. The model itself is never modified.

Start with the [branching example](examples/svg/branching.after.svg) or the TinyLlama
[before](examples/svg/tinyllama.focus.before.svg) and [after](examples/svg/tinyllama.focus.after.svg)
detail views. Download SVGs and open them in a browser if GitHub cannot preview their size.

## Install and run

Use Python 3.11–3.13. Install the **Graphviz system executable** as well as the Python dependencies.
On macOS use `brew install graphviz`; on Debian/Ubuntu use `sudo apt install graphviz`.

From the cloned repository:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
dot -V

model-graph-viewer --model examples/models/mlp.xml --output graph.svg
model-graph-viewer --model examples/models/branching.xml --output simplified.svg \
  --passes hide-constants,highlight --highlight projection --stats stats.json
```

The small models are included; no network model download is needed for this quick start or the tests.
`requirements.lock` pins the runtime and development packages used for verification. For an unpinned
development installation, `python -m pip install -e '.[dev]'` is also supported. The separate optional
`inspection` extra is only needed to reproduce the Netron screenshot.

## CLI

`python -m model_graph_viewer` and `model-graph-viewer` use the same API.

| Option | Behavior |
| --- | --- |
| `--model PATH` | OpenVINO `.xml`; required weights are read from its adjacent `.bin` |
| `--output PATH` | Exact destination filename, ending in `.svg`; parent directories are created |
| `--passes LIST` | Comma-separated pass names, executed in this exact order; default is no passes |
| `--highlight NAME` | Exact name for the highlighting pass; repeat for multiple names |
| `--stats PATH` | Write before/after graph sizes and stage times as JSON |
| `--render-timeout SECONDS` | Maximum Graphviz execution time, default 300 seconds |

The supported passes are:

- `highlight`: red borders on **all** matching names, including duplicates. Other nodes are unchanged.
- `hide-constants`: replace Constant connections with destination-port annotations such as
  `[1]: f32[4, 4] (shared: 2)`. Shared IDs identify the same Constant across consumer operations.
- `hide-constant-converts`: summarize direct `Constant → Convert` inputs, preserving stored/effective
  dtypes, effective shape, source ID, and conversion ID. Run this **before** `hide-constants`.

Highlighting is opt-in and requires at least one `--highlight`. Unmatched names do nothing.
Selecting no passes produces the original topology. Unknown passes, invalid inputs, missing weights,
and rendering failures produce a nonzero exit code and an error on stderr. Failed SVG rendering leaves
an existing output intact. SVG and statistics paths cannot alias the input model or its weights,
including symbolic links and hard links. Statistics are written atomically.

## Python API

Use the individual stages when inspecting or modifying the graph:

```python
from model_graph_viewer import (
    HideConstants, HighlightNodes, apply_passes, load_model, render_svg,
)

graph = load_model("examples/models/branching.xml")
original = graph.copy()  # optional: preserve a separate visualization variant
apply_passes(graph, [HideConstants(), HighlightNodes(["projection"])])
render_svg(original, "original.svg")
render_svg(graph, "simplified.svg", timeout=300)
```

Or use the same convenience function as the CLI:

```python
from model_graph_viewer import HideConstants, visualize

result = visualize(
    "examples/models/mlp.xml", "mlp.svg",
    passes=[HideConstants()], render_timeout=300,
)
print(result.before_nodes, result.after_nodes, result.timings)
```

`VisualizationResult` includes `output_path`, `before_nodes`, `before_edges`, `after_nodes`,
`after_edges`, and `timings`. Timing keys are `load_construct_seconds`, `transform_seconds`, and
`render_seconds`. All use wall-clock seconds. Passes mutate and return the detached graph; they do
not retain or mutate OpenVINO objects.

## Architecture and extension points

The pipeline is **OpenVINO adapter → graph → ordered passes → Graphviz renderer**.
See [the design walkthrough](docs/design.md) for invariants, complexity, tradeoffs, and the optional pass.

| Component | Responsibility |
| --- | --- |
| `graph.py` | Dataclasses, node identity, port tables, indexed connectivity, safe mutation, validation |
| `openvino_loader.py` | Read IR; translate framework operations/ports/metadata into ordinary Python data |
| `passes.py` | In-place transformations and pass registration; no framework dependencies |
| `render.py` | Escaped HTML labels, port attachment points, styles, bounded large-graph layout, SVG output |
| `api.py` / `cli.py` | Compose the stages, collect timings, parse CLI configuration, report failures |

Node IDs are separate from names; edges include both port indices. Input/output port tables survive
hiding connections. Shapes distinguish unavailable metadata (`None`), dynamic rank (`"?"`), scalars
(`()`), and dimensions such as `(1, "?", "2..8", "2..?")`. Dtypes use OpenVINO's portable names,
such as `f16`, `f32`, and `i64`. The core also permits absent dtypes.

Add a pass by implementing `apply(graph) -> graph`; no parser or renderer changes are needed:

```python
from model_graph_viewer import Graph, apply_passes
from model_graph_viewer.passes import create_pass, register_pass

class HighlightType:
    def __init__(self, operation="MatMul"):
        self.operation = operation

    def apply(self, graph: Graph) -> Graph:
        for node in graph.nodes.values():
            if node.node_type == self.operation:
                node.visual.border_color = "blue"
        return graph

register_pass("highlight-type", HighlightType)
# With a graph loaded as above:
apply_passes(graph, [create_pass("highlight-type", operation="MatMul")])
```

The registry accepts configurable factories. CLI integration for new pass-specific flags belongs in
the CLI; existing loaders and rendering remain unchanged. Built-in structural passes have no extra
options. Duplicate registrations are rejected.

To support another format, implement `load(path) -> Graph` following the `ModelLoader` protocol and
pass the instance to `load_model(path, loader=...)` or `visualize(..., loader=...)`. Populate explicit
port tables and detached tensor metadata, then validate the graph. Existing passes and rendering
work without knowing the originating framework. ONNX loading is an extension point, not an
implemented feature.

## Models and reproducible visualizations

| Model | What it demonstrates | SVGs |
| --- | --- | --- |
| MLP | Starter's MatMul/Add structure with a corrected `(1, 1024)` bias | [Before](examples/svg/mlp.before.svg), [after](examples/svg/mlp.after.svg) |
| Branching | Dynamic batch, Split outputs, shared weights, duplicate `projection` names, repeated node-pair connections | [Before](examples/svg/branching.before.svg), [after](examples/svg/branching.after.svg) |
| TinyLlama-1.1B-Chat-v1.0 | Real large, stateful fp16 model and converted-weight simplification | [Before](examples/svg/tinyllama.before.svg), [after](examples/svg/tinyllama.after.svg) |

The first two are deterministic, generated neural-network examples, not pretrained models.
The generator uses fixed random seeds; its XML and weights were checked for byte-for-byte reproduction.

```sh
python scripts/build_examples.py
python scripts/benchmark.py --models small
```

TinyLlama uses the official [OpenVINO fp16 export](https://huggingface.co/OpenVINO/TinyLlama-1.1B-Chat-v1.0-fp16-ov),
pinned to revision `88a9e6bbe7f9f8cf4e61ac96eed57ec164ad46f5`. The exact source, sizes, and SHA-256
digests are in [model_sources.json](examples/model_sources.json). Downloading requires `curl` and
about 2.2 GB of disk space. Downloads verify HTTPS, resume partial files, and check both hashes.
Large weights are ignored by Git.

```sh
python scripts/download_tinyllama.py
model-graph-viewer --model models/tinyllama/openvino_model.xml --output tinyllama.svg \
  --passes hide-constant-converts,hide-constants
python scripts/render_focus.py
python scripts/benchmark.py --models all --include-synthetic --repeats 1
```

The final command reproduces all benchmarked SVGs and overwrites `benchmarks/results.json` with new
measurements. Use `--output PATH` and `--svg-directory PATH` to preserve the supplied results.
`--repeats N` retains every measured run in JSON; each variant's SVG is replaced on each repetition.

### Optional improvement and Netron inspection

The pinned export contains 428 direct Constant-to-Convert branches. In Netron, Constant inputs are
already embedded in operation boxes, but the conversion nodes and their connections remain. The
[captured view](docs/netron-tinyllama.png) shows repeated Convert branches around the main arithmetic
and many crossing connections. The optional pass removes these auxiliary display nodes while keeping
their effective tensor description and conversion provenance at the consumer.

The [focused before](examples/svg/tinyllama.focus.before.svg) and
[focused after](examples/svg/tinyllama.focus.after.svg) SVGs show the first attention query projection:
five nodes/four edges become three nodes/two edges. Boundary connections omitted from this detail view
are explicitly labeled; the full SVGs include the complete model graph.

The Netron screenshot examines the original XML topology without loading tensor values:

```sh
python -m pip install -e '.[inspection]'
python -m playwright install chromium
python scripts/inspect_netron.py
```

This optional script uses a temporary XML-only copy, localhost, and a fresh browser; external telemetry
requests are blocked. To explore manually, open the pinned XML in Netron and inspect the first layer's
normalization and query-projection branches.

## Measured performance

Measured on Apple M4, 10 logical CPUs, 16 GiB RAM, macOS 27.0.1 arm64; Python 3.12.14,
OpenVINO 2026.0.0, NumPy 2.4.6, Python Graphviz 0.21, and Graphviz `dot` 16.1.0.
Exact environment, commands, input hashes, layout settings, SVG counts, and raw wall times are in
[results.json](benchmarks/results.json).

| Model / variant | Rendered nodes / edges | Load + construct (s) | Passes (s) | Render + export (s) |
| --- | ---: | ---: | ---: | ---: |
| MLP / baseline | 6 / 5 | 0.13956 | 0.00003 | 0.10631 |
| MLP / improved | 4 / 3 | 0.00386 | 0.00013 | 0.09676 |
| Branching / baseline | 9 / 11 | 0.00449 | 0.00004 | 0.10006 |
| Branching / improved | 7 / 8 | 0.00430 | 0.00018 | 0.10559 |
| TinyLlama / baseline | 3,216 / 3,789 | 0.41779 | 0.01216 | 15.08915 |
| TinyLlama / improved | 1,529 / 2,028 | 0.41075 | 0.04766 | 6.58634 |
| Synthetic stress / improved | 5,100 / 5,050 | 0.04115 | 0.06840 | 4.35199 |

These are single runs, not averages or performance guarantees. Each variant independently reloads
its model in the same process. The first load includes lazy imports; later runs may benefit from OS
caches. Pass time includes validation. Render time includes DOT creation, layout, and disk output.
The synthetic case constructs a detached graph with 5,101 nodes/5,100 edges; its construction time
does **not** measure OpenVINO loading. One shared Constant is hidden, leaving over 5,000 rendered nodes.

SVG XML was parsed to verify exact node and edge counts against each measured graph. The full graph
is retained in every baseline/improved example. The synthetic SVG is also included as a reproducible
[stress artifact](examples/svg/synthetic_5101.after.svg).

Default Graphviz layout proved impractical for the full TinyLlama graph; the interrupted attempt is
recorded separately in [interrupted_attempt.json](benchmarks/interrupted_attempt.json), without an
invented completion time. The final renderer uses bounded optimization and straight edge routing
above 1,000 nodes. This made layout practical but long edges can cross unrelated nodes. Graph
processing is inexpensive relative to layout; focused views are substantially easier to read.

## Tests and package verification

```sh
python -m pytest -q
ruff check .
ruff format --check src tests scripts starter_code.py
python -m build --wheel
```

The suite covers independent graph/pass behavior, shared constants, preserved port indices, shapes,
duplicate names, mutation validity, pass ordering, actual OpenVINO parsing, unchanged model metadata,
and both API/CLI end-to-end rendering. SVG tests parse structure and styles, not pixel snapshots.
Rendering failure tests cover missing `dot`, invalid output, label escaping, and timeout cleanup.
Tests use local models and require no network access or TinyLlama download.

Clean wheel installations were checked outside the source directory; exact Python versions, test
outcomes, package origins, and CLI checks are recorded in [verification.json](docs/verification.json).
The [requirements audit](docs/requirements-audit.md) maps the PDF's requirements to implementation,
tests, examples, and measured evidence, and identifies the remaining GitHub publication step.

## Known limitations

- OpenVINO IR is the only implemented model format. Model loading may map/read weights internally;
  the detached graph and transformations never copy their tensor contents.
- Nested If/Loop bodies remain represented by their parent operation. Only tensor/data connections
  are drawn; control-ordering dependencies are not additional edges. ReadValue/Assign sinks and
  variable IDs are preserved.
- Dynamic bounds are preserved, but relationships between framework-specific symbolic dimensions
  are not. Missing metadata is displayed explicitly rather than guessed.
- The optional pass handles only direct Constant-to-Convert patterns. Its documented order matters,
  and it deliberately does not evaluate tensors or simplify arbitrary runtime conversions.
- Large SVGs are static, tall, and potentially slow to navigate. Straight routing trades readability
  for bounded layout cost. Graphviz versions/fonts may change geometry; exact SVG bytes are not a
  cross-platform contract. Neighborhood selection and hierarchical block views are useful extensions.

`starter_code.py` remains runnable as the supplied starting example, with its bias shape corrected.
The package extends its dataclass/Graphviz approach with explicit port tables, transformations,
framework separation, and rendering that tolerates operation types not known in advance.
