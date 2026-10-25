# Design walkthrough

## Separate model, graph, and presentation

The OpenVINO adapter reads operations and tensor connections once and returns ordinary Python data.
Passes and rendering receive no OpenVINO nodes, tensor buffers, or runtime handles. Hiding weights,
choosing reference connections, and changing label detail therefore cannot change the executable model.
The transformed graph is a visualization representation, not an executable IR.

`Node.node_id` identifies an operation. `Node.name` can repeat and is never used as identity.
IDs follow topological order and are repeatable for the same serialized model and dependency version;
they are not global identifiers across different exports. JSON uses string object keys for those IDs.

Port tables remain complete when connections are hidden or referenced. `Split:out 1 → Add:in 0` and
`Split:out 0 → Add:in 1` are distinct connections even when the operation pair is identical.
Runtime positional indices are used; XML output-port IDs can follow the input IDs and cannot be used
as output indices directly. Scalars, dynamic rank, bounded dimensions, and unavailable shapes remain distinct.

## Connectivity and transformation cost

The graph indexes operation IDs, connections, incoming/outgoing adjacency, and each input's producer.
Adding/removing a connection takes expected constant time; removing an operation takes time proportional
to its incident connections. Graph construction, validation, and each built-in pass take O(V + E).
Node/edge collections are read-only views, while attributes and visualization metadata remain mutable.

An input has at most one producer. Multiple connections between an operation pair are valid when their
destination ports differ. Isolated operations, unused outputs, disconnected inputs, and stateful sinks
are preserved. The core permits cycles because it represents connectivity rather than framework execution.

`apply_passes` mutates and validates the supplied graph in the requested order. It does not copy a graph
per pass. `graph.copy()` explicitly creates an independent variant. The convenience API records one
portable original metadata snapshot before passes, then exports the resulting view. That snapshot costs
O(V + E) additional metadata and is needed to retain names/attributes of operations removed by hiding.
It never contains tensor values or another framework model.

## Constant hiding and conversion provenance

`HideConstants` annotates each consumer's original input index with tensor dtype and shape, then
removes the hidden connection. A Constant is removed only when no visible connection remains.
Unrelated connections survive unchanged. A shared marker identifies the Constant by operation ID,
not by name; sharing means multiple distinct consumer operations, rather than multiple ports on one
consumer. Provenance is recorded before branches are removed, making repeated application safe.

The adapter never calls Constant data accessors or `Constant.get_attributes()`, which can expose
values. Tensor descriptions come from ports. Neither passes nor metadata export evaluates or copies weights.

The pinned fp16 TinyLlama export has 428 direct `Constant → Convert` branches. The original
[Netron inspection](netron-tinyllama.png) showed these auxiliary conversions around the main arithmetic.
`HideConstantConverts` summarizes direct, single-input/single-output conversions at their consumers.
Annotations retain stored dtype, effective tensor dtype/shape, source Constant ID, and Convert ID.
Distinct casts of one source remain distinct effective values. This is display simplification, not folding.

Run converted-weight hiding before ordinary Constant hiding. In reverse order the direct Constant
pattern is no longer visible, so the Convert stays. Longer conversion chains remain partly visible.
The original Constant/Convert detail preserves this demonstration with explicit boundary captions.

## Why fan-out references

Constant/Convert hiding reduced TinyLlama to 1,529 operations and 2,028 logical connections, but the
previous SVG was still 12,038 × 170,016 points. Names and verbose attributes widened operation boxes;
shared non-Constant values produced long connections across decoder layers. Eight source operations
had more than three outgoing connections, accounting for 250 connections. Five each had 44.

`ReferenceHighFanout(threshold=3)` marks every connected output of a source with more than three
outgoing logical connections. Counting is per operation, across outputs, and includes separate inputs
on the same consumer. It updates `visual.reference_outputs`; no operation or logical edge is deleted.
The pass clears and recomputes its markers, so it is idempotent and reflects earlier structural passes.
Run it after hiding to select fan-out from the surviving graph. Pass order is never silently rearranged.

The renderer omits each marked edge path and displays two matching pieces of information:

- Source output: `out 0 → @40:0 (4 uses)`.
- Consumer input: `in 0 ← @40:0`, linked to SVG element `node-40`.

The token contains both operation ID and output port. Consumer placement supplies the exact input
index, so multiple outputs, repeated operation pairs, and multiple inputs remain unambiguous. Tooltips
state the full directed endpoint tuple. Source markers report counts per output, while the threshold
is applied to the operation's total outgoing count. References need no framework-specific semantics.

Reference connections do not constrain Graphviz ranks. This removes their long layout chains, but
can place their sources in separate components. Following a token or source link replaces tracing a
long line. Every logical connection is also present in `view.edges` with its representation recorded.
The small fan-out demonstration retains all four consumers of a real source and labels its omitted
external boundary connections. It does not claim to be the full model.

## Compact labels and metadata

Compact mode displays operation type, ID, exact ports, short scalar attributes, and all required hidden
Constant annotations. A scalar attribute is displayed when its complete `key: value` text is at most
40 characters. Names and verbose attributes are omitted from labels; names remain available in hover
tooltips. Full mode displays names and all ordinary attributes. Unknown operations use neutral styling.
Text wrapping limits long rows without truncating their content. Highlighting remains a red border.

Every new render writes schema-version-1 JSON. Root `nodes` and `edges` describe the original graph;
`view.nodes` and `view.edges` describe the transformed graph. Edges retain exact source/output and
destination/input indices plus tensor metadata. View edges add `representation: "edge"|"reference"`.
Node records include full ordinary attributes, port tables, and visualization metadata, including hidden
Constant provenance and reference-output markers. `rendering` records label detail and edge counts.
Shapes encode unavailable metadata as `null`, dynamic rank as `"?"`, and a scalar as `[]`.

`visualize` takes the original snapshot before applying passes. Low-level `render_svg` describes the
graph supplied to it. Focus examples deliberately snapshot their prepared neighborhood, not the full
model; the full-model exports retain the complete original snapshot. Snapshot records are detached
from subsequent graph mutations.

## Layout, output handling, and independent checks

HTML tables supply named port attachment points. Labels are escaped for HTML and Graphviz backslash
substitution; XML-forbidden characters are replaced with U+FFFD. Source links and reference IDs use
numeric operation/port identities, including supported negative IDs.

An earlier unrestricted TinyLlama layout generated 422,151 internal positioning nodes and 631,894
positioning edges for 3,216 actual operations. Graphs above 1,000 operations therefore limit Graphviz
optimization with `nslimit=2`, `nslimit1=2`, and `mclimit=0.1`.

Small graphs use polyline routing. Large graphs with reference connections also use polylines to route
around unrelated operation boxes. Large graphs with no reference connections use straight lines:
unreduced polyline routing remained expensive, so complete original/compact comparisons use the bounded
fallback. Those straight lines can cross unrelated boxes. Every benchmark run records the actual
layout settings; historical straight-routing timings without JSON are not treated as equivalent measurements.

The renderer writes both outputs to temporary files, then replaces the destinations. Failed generation
keeps an existing pair intact; replacement errors trigger rollback where possible. Two filesystem
replacements cannot be one transaction if the process is interrupted between them. Graphviz has a
configurable timeout. Missing executables, invalid output, and layout failures produce actionable errors.

The API/CLI reject any SVG, metadata, or statistics destination that aliases the input XML/BIN or
another output, including symbolic links and hard links. Statistics are written atomically after the
visualization succeeds. A statistics failure can leave a valid completed SVG/JSON pair.

After timing, the benchmark independently reads SVG operation IDs, exact directed edge tooltips,
reference IDs/links/text, and source tokens/use counts. It reconstructs every logical endpoint tuple,
checks its representation, and compares it against the graph and JSON snapshots. It also verifies
operation attributes and tensor metadata, then measures canvas dimensions and operation box widths.
This detects omitted/duplicated connections and misattached references that aggregate counts could
miss. Drawn endpoints are read from tooltips; separate unit tests check the exact DOT port attachments.

## Deliberate boundaries

Only data connections are represented. ReadValue/Assign operations and variable IDs remain, but extra
control-ordering dependencies are not synthesized. Nested If/Loop bodies stay within their parent
operation. Dynamic dimension bounds are retained, but symbolic equality relationships are not.

Large SVGs remain static and tall. References are a navigation tradeoff, not a claim that every whole
model becomes readable at once. Focused neighborhoods, decoder-block grouping, and interactive
hierarchical views remain useful extensions without changing the loader/pass separation.
