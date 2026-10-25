# Model Graph Viewer

[![Tests](https://github.com/agastya-kataria/model-graph-viewer/actions/workflows/ci.yml/badge.svg)](https://github.com/agastya-kataria/model-graph-viewer/actions/workflows/ci.yml)

A Python library and CLI for visualizing OpenVINO computational graphs as SVGs. It preserves
operation identity, tensor metadata, and exact port connections while applying ordered visualization
passes. The original model is never modified.

Start with the TinyLlama fan-out detail [before](examples/svg/tinyllama.fanout.focus.before.svg) and
[after](examples/svg/tinyllama.fanout.focus.after.svg), then the
[complete improved graph](examples/svg/tinyllama.after.svg). Download SVGs and open them in a browser
to use the source links. Each new render also writes JSON containing names, attributes, and connectivity.

## Install and run

Use Python 3.11–3.13. Install the **Graphviz system executable** as well as the Python dependencies:
`brew install graphviz` on macOS, or `sudo apt install graphviz` on Debian/Ubuntu.

```sh
git clone https://github.com/agastya-kataria/model-graph-viewer.git
cd model-graph-viewer
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
dot -V

# No passes: original topology, compact labels, graph.json metadata.
model-graph-viewer --model examples/models/mlp.xml --output graph.svg

# Select transformations explicitly; their order is preserved.
model-graph-viewer --model examples/models/branching.xml --output simplified.svg \
  --passes hide-constants,highlight --highlight projection --stats stats.json
```

The two small models are included; quick-start commands and tests need no model download.
`requirements.lock` pins the verified runtime and development dependencies. An unpinned development
installation with `python -m pip install -e '.[dev]'` is also supported.

## Readability changes

The earlier TinyLlama result removed many auxiliary Constant/Convert nodes, but still contained long
names, wide attribute rows, and shared values connected to operations across many decoder layers.
Eight surviving source operations have more than three outgoing connections, accounting for 250 of
the simplified graph's 2,028 connections. Five of those sources each have 44 outgoing connections.
These long connections enlarge the layout and make the main computation difficult to follow.

The revised visualization addresses those problems in three ways:

- Compact labels omit node names and verbose attributes, retain operation types and IDs, and display
  scalar attributes whose complete `key: value` text is at most 40 characters. Hidden Constant dtype,
  shape, original input index, and sharing markers remain visible. JSON retains full metadata.
- `reference-high-fanout` replaces drawn connections from qualifying sources with matching port
  references. `out 0 → @40:0 (4 uses)` declares a source; `in 0 ← @40:0` identifies its use at an exact
  consumer input. Click a reference to jump to its source; browser Back returns to the consumer.
  Every logical connection remains in the graph and JSON.
- Large graphs with references use polyline routing around unrelated operation boxes. Graphs above
  1,000 operations without references retain straight routing because unreduced polyline layout is
  expensive. Large graphs also bound Graphviz optimization; each benchmark records its actual settings.

The fan-out pass counts outgoing **connections per operation**, across its output ports. The default
threshold is three: exactly three connections stay drawn; four or more are referenced. Multiple
connections to one consumer count separately. Each output uses its own token, so a multi-output
operation's values remain distinct. Run this pass after structural hiding passes to select sources
using the remaining graph. Reapplying it recomputes markers without changing logical connectivity.

The [fan-out detail](examples/svg/tinyllama.fanout.focus.after.svg) uses real node 40 and all four of
its consumers. Both versions retain five operations and four logical connections; only their visual
representation changes. A visible caption identifies six omitted external boundary connections.
The detail JSON describes that prepared neighborhood; full-model JSON describes the complete model.

## CLI

`python -m model_graph_viewer` and `model-graph-viewer` use the same public API.

| Option | Behavior |
| --- | --- |
| `--model PATH` | OpenVINO `.xml`; adjacent `.bin` is read when required |
| `--output PATH` | SVG destination; parent directories are created |
| `--passes LIST` | Comma-separated names, executed in order; default: no passes |
| `--highlight NAME` | Exact name for `highlight`; repeat for multiple names |
| `--fanout-threshold N` | Positive integer; requires `reference-high-fanout`; default: 3 |
| `--label-detail compact\|full` | Default: compact; full includes names and all ordinary attributes |
| `--metadata PATH` | JSON destination; default: SVG path with suffix changed to `.json` |
| `--stats PATH` | Separate JSON with logical/drawn/referenced counts and stage timings |
| `--render-timeout SECONDS` | Maximum Graphviz execution time; default: 300 seconds |

Supported passes:

- `highlight`: red borders on all exact name matches, including duplicates; other nodes are unchanged.
- `hide-constants`: replace Constant connections with consumer input annotations such as
  `[1]: f32[4, 4] (shared: 2)`. The shared ID identifies the same Constant across consumers.
- `hide-constant-converts`: summarize direct `Constant → Convert` branches while preserving source
  and conversion IDs, stored/effective dtypes, and effective shape. Run before `hide-constants`.
- `reference-high-fanout`: mark high-fan-out source outputs for reference rendering; no nodes or
  logical edges are deleted.

```sh
model-graph-viewer --model models/tinyllama/openvino_model.xml --output tinyllama.svg \
  --passes hide-constant-converts,hide-constants,reference-high-fanout \
  --fanout-threshold 3 --metadata tinyllama.metadata.json --stats tinyllama.stats.json

# Full labels without transformation passes.
model-graph-viewer --model examples/models/mlp.xml --output full.svg --label-detail full
```

Highlighting requires at least one name; unmatched names do nothing. A threshold option requires the
fan-out pass. Invalid arguments, unsupported models, missing weights, and rendering failures return
nonzero status with an error on stderr. SVG, metadata, and statistics destinations must be distinct
and cannot alias input XML/BIN files, including symbolic links and hard links. The SVG/JSON pair is
staged before replacement; failed generation preserves an existing pair. Statistics are written
atomically after visualization succeeds.

## Python API and metadata

```python
from model_graph_viewer import (
    HideConstantConverts, HideConstants, ReferenceHighFanout, visualize,
)

result = visualize(
    "models/tinyllama/openvino_model.xml", "tinyllama.svg",
    passes=[HideConstantConverts(), HideConstants(), ReferenceHighFanout(threshold=3)],
    label_detail="compact", render_timeout=300,
)
print(result.after_edges, result.rendered_edges, result.referenced_edges)
print(result.metadata_path, result.timings)
```

`VisualizationResult` contains `output_path`, `metadata_path`, before/after logical node and edge
counts, `rendered_edges`, `referenced_edges`, and wall-clock stage timings. Timing keys are
`load_construct_seconds`, `transform_seconds`, and `render_seconds`. References reduce drawn edges;
they do not reduce `after_edges`.

Use `load_model`, `apply_passes`, and `render_svg` separately when inspecting the graph. Passes mutate
the detached graph in place. `graph.copy()` creates an independent variant. `render_svg` also accepts
`label_detail`, `metadata`, and `timeout`, and returns the SVG path. Its metadata describes the graph
supplied to it. Use `visualize` to retain the original model graph before transformations as well.

The JSON schema has `schema_version: 1` and these fields:

| Field | Meaning |
| --- | --- |
| `nodes` | Original nodes keyed by string node ID, including names, attributes, ports, and visualization metadata |
| `edges` | Original exact source/output/destination/input connections and tensor metadata |
| `view.nodes` | Nodes retained after passes, including hidden Constant annotations and reference-output markers |
| `view.edges` | All remaining logical connections, each with `representation: "edge"` or `"reference"` |
| `rendering` | Label detail and counts of drawn and referenced edges |

IDs remain integers inside node/edge records; JSON object keys are strings. Shape `null` means
unavailable metadata, `"?"` means dynamic rank, and `[]` means a scalar. No tensor values are exported.
Names remain available in SVG tooltips without affecting compact label width.

```python
import json

with open("tinyllama.json") as stream:
    metadata = json.load(stream)
node_id = 40  # Read the ID from the SVG.
print(metadata["nodes"][str(node_id)]["name"])
references = [edge for edge in metadata["view"]["edges"]
              if edge["representation"] == "reference" and edge["from_node"] == node_id]
```

View edge endpoints use the same operation IDs as the original mapping. `view.nodes` additionally
records the transformed display state; original nodes removed by hiding remain in root `nodes`.

## Architecture and extension points

The pipeline is **OpenVINO adapter → detached graph → ordered passes → Graphviz SVG + metadata**.
The [design walkthrough](docs/design.md) explains invariants, complexity, and tradeoffs.

| Component | Responsibility |
| --- | --- |
| Graph | Operation identity, indexed port-aware connectivity, validated mutation, visualization metadata |
| OpenVINO adapter | Translate operations, attributes, ports, and tensor metadata into ordinary Python data |
| Passes | Independent, ordered graph transformations and reference-selection policy |
| Renderer / metadata | Label detail, directed connections/references, styles, SVG and original/view snapshots |
| API / CLI | Compose stages, validate destinations, configure passes, collect timings, report errors |

IDs are separate from names. Runtime positional ports are preserved, including parallel connections
between the same node pair. Dynamic bounds, unavailable metadata, unused ports, disconnected model
inputs, stateful sinks, and unknown operation types are supported.

Add a pass by implementing `apply(graph) -> graph`; no framework objects are required:

```python
from model_graph_viewer import Graph
from model_graph_viewer.passes import register_pass

class HighlightType:
    def __init__(self, operation="MatMul"):
        self.operation = operation

    def apply(self, graph: Graph) -> Graph:
        for node in graph.nodes.values():
            if node.node_type == self.operation:
                node.visual.border_color = "blue"
        return graph

register_pass("highlight-type", HighlightType)
```

The registry accepts configurable factories; duplicate registrations are rejected. Add CLI flags for
new pass-specific options in the CLI. Existing parsers and generic rendering remain independent of
the pass. To support another model format, implement `load(path) -> Graph` following `ModelLoader`
and supply it to `load_model(..., loader=...)` or `visualize(..., loader=...)`. Populate port tables
and detached metadata, then validate the graph. ONNX is an extension point, not an implemented loader.

## Models and reproduction

| Model | Demonstrates | SVG variants |
| --- | --- | --- |
| MLP | Starter MatMul/Add with correctly broadcast `(1, 1024)` bias | [Original](examples/svg/mlp.before.svg), [compact](examples/svg/mlp.compact.svg), [improved](examples/svg/mlp.after.svg) |
| Branching | Dynamic batch, Split outputs, shared weights, duplicate names, repeated connections | [Original](examples/svg/branching.before.svg), [compact](examples/svg/branching.compact.svg), [improved](examples/svg/branching.after.svg) |
| TinyLlama-1.1B-Chat-v1.0 | Real stateful fp16 model, converted weights, long shared connections | [Original](examples/svg/tinyllama.before.svg), [compact](examples/svg/tinyllama.compact.svg), [improved](examples/svg/tinyllama.after.svg) |

The first two are deterministic generated examples, not pretrained networks. XML/BIN files are
included. They have no surviving fan-out above three, so their compact and improved variants agree.

TinyLlama uses the official [OpenVINO fp16 export](https://huggingface.co/OpenVINO/TinyLlama-1.1B-Chat-v1.0-fp16-ov)
at revision `88a9e6bbe7f9f8cf4e61ac96eed57ec164ad46f5`. Source URLs, sizes, and SHA-256 digests are
in [model_sources.json](examples/model_sources.json). The downloader requires `curl` and about 2.2 GB
of disk space; it resumes partial downloads and verifies both files. Weights are ignored by Git.

```sh
python scripts/build_examples.py
python scripts/download_tinyllama.py
python scripts/benchmark.py --models all --include-synthetic --repeats 1
python scripts/render_focus.py
python scripts/render_fanout_focus.py
```

These commands reproduce three variants per real model, JSON sidecars, both detail views, the
5,100-node synthetic stress SVG, and benchmark measurements. The original Constant/Convert detail
[before](examples/svg/tinyllama.focus.before.svg)/[after](examples/svg/tinyllama.focus.after.svg) remains
available. Each detail explicitly identifies its omitted boundary connections.

The benchmark overwrites `benchmarks/results.json` and generated SVG/JSON files. Use `--output PATH`
and `--svg-directory PATH` to preserve supplied results. `--repeats N` records every run, while each
variant's files are replaced on each repetition. The earlier submitted TinyLlama visualization is
preserved as [tinyllama.previous.svg](examples/svg/tinyllama.previous.svg), a historical comparison
artifact rather than output from the current renderer.

The [Netron screenshot](docs/netron-tinyllama.png) documents the original inspection. Its Constant-to-
Convert branches motivated the earlier optional pass; the pinned model has 428 direct branches.
To reproduce that inspection, install `'.[inspection]'`, run
`python -m playwright install chromium`, then `python scripts/inspect_netron.py`.

## Measured performance and verification

[results.json](benchmarks/results.json) records actual load/construction, snapshot/transformation,
and SVG/JSON export times; graph counts; canvas geometry; operation widths; file sizes; commands;
dependency versions; and hardware. Real-model variants separate full original topology, compact
Constant/Convert hiding, and the additional effect of fan-out references. The synthetic stress fixture
retains 5,100 operations after hiding one Constant; it is not an OpenVINO loading benchmark.

Measured on Apple M4, 10 logical CPUs, 16 GiB RAM, macOS 27.0.1 arm64, Python 3.12.14,
OpenVINO 2026.0.0, NumPy 2.4.6, Python Graphviz 0.21, and Graphviz `dot` 16.1.0:

| Model / variant | Operations / logical edges | Drawn / referenced edges | Load (s) | Snapshot + passes (s) | SVG + JSON export (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLP / original | 6 / 5 | 5 / 0 | 0.08671 | 0.00009 | 0.08495 |
| MLP / compact | 4 / 3 | 3 / 0 | 0.00238 | 0.00011 | 0.06200 |
| MLP / improved | 4 / 3 | 3 / 0 | 0.00222 | 0.00012 | 0.06599 |
| Branching / original | 9 / 11 | 11 / 0 | 0.00257 | 0.00009 | 0.06307 |
| Branching / compact | 7 / 8 | 8 / 0 | 0.00266 | 0.00016 | 0.05731 |
| Branching / improved | 7 / 8 | 8 / 0 | 0.00264 | 0.00019 | 0.07189 |
| TinyLlama / original | 3,216 / 3,789 | 3,789 / 0 | 0.22517 | 0.02803 | 9.00939 |
| TinyLlama / compact | 1,529 / 2,028 | 2,028 / 0 | 0.23134 | 0.05723 | 3.98579 |
| TinyLlama / improved | 1,529 / 2,028 | 1,778 / 250 | 0.20849 | 0.05072 | 1.01339 |
| Synthetic stress / improved | 5,100 / 5,050 | 5,050 / 0 | 0.02266 | 0.10497 | 3.71383 |

| TinyLlama visualization | Canvas width × height (pt) | Median operation width (pt) | Drawn edges |
| --- | ---: | ---: | ---: |
| Previous submission | 12,038 × 170,016 | 259.75 | 2,028 |
| Current original | 20,671 × 145,845 | 219.25 | 3,789 |
| Compact | 6,542 × 148,862 | 109.00 | 2,028 |
| Improved with references | 1,975 × 152,284 | 109.00 | 1,778 |

Compared with the earlier submission, the improved graph is 83.6% narrower and 10.4% shorter.
Compared with compact labels alone, references make the canvas 69.8% narrower but 2.3% taller.
The graph remains tall; the improvement concerns width, long shared connections, and source navigation.
The improved files contain 4,164,347 SVG bytes and 5,394,119 JSON bytes. Metadata includes both original
and transformed graphs, so its size is an explicit cost of retaining complete context.

Each run independently loads its model in one process without warm-up. First-use imports and OS
caching affect timings. Transformation time includes the original metadata snapshot and validation;
render time includes DOT generation, Graphviz layout, and both files. These are individual
measurements, not performance guarantees. Package verification was also active on the machine;
shared CPU/I/O can affect wall times. Graphviz layout remains the main bottleneck.

After timing, the benchmark independently recovers exact endpoints from drawn-edge tooltips and
reference IDs/links, verifies source markers/use counts, and compares every logical edge with the JSON
snapshots. It also compares node identities and tensor metadata. This verifies connectivity instead
of assuming the number of drawn edges equals the number of graph connections.

Historical timings in the earlier submission used straight routing and no JSON export;
they are not directly comparable to the current benchmark. Geometry varies with Graphviz and fonts.

## Tests

```sh
python -m pytest -q
ruff check .
ruff format --check src tests scripts starter_code.py
python -m build --wheel
```

Tests cover graph invariants, parsing, shared Constants, conversion provenance, fan-out thresholds,
multiple source/destination ports, duplicate names, shapes, ordering, idempotence, SVG references,
metadata snapshots, escaping, output protection, and API/CLI integration. Small real OpenVINO models
exercise the whole pipeline without network access. Failure tests cover missing `dot`, timeouts,
invalid destinations, and output staging. No pixel-exact SVG snapshots are required.

GitHub Actions tests built packages on Python 3.11–3.13 and checks linting/formatting.
[verification.json](docs/verification.json) records installed-wheel verification;
the [requirements audit](docs/requirements-audit.md) maps the assignment and revision to evidence.
The recorded clean-wheel matrix passed all 230 tests on each of Python 3.11, 3.12, and 3.13.
To reproduce the clean-wheel matrix outside the source tree, run `python scripts/verify_package.py`.
It requires `uv` and available or downloadable Python 3.11–3.13 interpreters, builds the wheel, installs
it into temporary environments, and runs the tests and CLI checks from outside the repository.

## Known limitations

- OpenVINO IR is the only input format. The runtime may read/map weights internally; detached graph
  processing never copies tensor values.
- Nested If/Loop bodies stay within their parent operation. Data connections are represented;
  additional control-ordering edges are not synthesized. Stateful ReadValue/Assign operations remain.
- Dynamic bounds are retained, but framework-specific symbolic equality relationships are not.
- Constant/Convert hiding handles direct branches only; it does not evaluate weights or bypass
  arbitrary runtime conversions. Its order relative to Constant hiding matters.
- References require matching tokens or source links instead of following every connection visually.
  They do not constrain layout ranks, so a source can appear in a separate component. All logical
  connections remain available through SVG references and JSON. Large static graphs remain tall;
  details are easier for local investigation.
- Large graphs without references use straight connections that can cross unrelated boxes. The
  reference-enabled layout uses obstacle-aware polylines, at the cost of a different layout policy.
- Output replacement failures are rolled back where possible; interruption between filesystem
  replacements cannot provide a cross-file transaction. Graphviz versions/fonts may change geometry.
  Decoder-block grouping and interactive hierarchy are possible extensions.

`starter_code.py` remains runnable as the supplied starting example, with its bias shape corrected.
