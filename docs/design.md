# Design walkthrough

## Why a separate graph?

The OpenVINO adapter reads operations and tensor connections once, then returns only Python data.
Passes and rendering do not receive the model, OpenVINO nodes, tensor buffers, or runtime handles.
This makes transformations independently testable and prevents an operation such as hiding a weight
from changing the executable model. The transformed graph is a **visualization**, not an executable IR.

`Node.node_id` identifies an operation. `Node.name` is display text and can repeat. The adapter assigns
IDs by topological order and uses OpenVINO instance identity to resolve connections. The IDs are
repeatable for the same serialized model and dependency version; they are not global identifiers
across different exports.

Every operation retains its complete input/output port tables. Removing an edge never renumbers
ports. For example, hiding MatMul input 1 leaves input 0 connected and an annotation at input 1.
`Split:out 1 → Add:in 0` is a different edge from `Split:out 0 → Add:in 1`, even though their node
pair is the same. Runtime positional indices are used: XML output-port IDs often follow the input IDs
and therefore cannot be used as output indices directly.

## Connectivity and mutation

The graph indexes nodes, edges, incoming/outgoing adjacency, and each input's producer. Adding and
removing a connection takes expected constant time; removing a node takes time proportional to its
incident edges. Construction, validation, and each built-in pass take O(V + E). Node and edge
collections are read-only views, while attributes and visualization metadata can be edited.

An input has at most one visible or hidden producer. Multiple connections between the same pair of
nodes remain valid when their destination ports differ. Isolated nodes, unused output ports, and
stateful sink operations are retained. The graph does not enforce DAG semantics because its job is
to represent and modify connections, not validate framework execution.

`apply_passes` mutates the supplied graph and validates each result. This avoids copying the entire
graph for every pass. `graph.copy()` explicitly copies mutable node state when independent variants
are needed; immutable edges and tensor descriptions can be shared safely.

## Constant hiding and sharing

For each Constant connection, `HideConstants` stores a `ConstantAnnotation` at the destination's
original input index. Its dtype and shape describe the hidden tensor. Operation attributes remain
separate, so an attribute with an identical textual name cannot overwrite the annotation.

Sharing means **multiple distinct consumer operations**, not multiple ports on one consumer.
The shared marker contains the unique source node ID, not its possibly duplicated name. Sharing is
recorded before removing any edges and retained when an earlier pass removes a branch. All relevant
connections are hidden before pruning isolated Constants, making repeated application safe.

The adapter never calls Constant data accessors or `Constant.get_attributes()`, which can expose
tensor values. Shape and dtype come from port metadata. No pass evaluates, copies, or displays weights.

## The TinyLlama improvement

The pinned fp16 export stores many weights as `Constant → Convert → operation`. Netron already
places Constant inputs inside operation boxes, but leaves the Convert nodes and their connections
visible. In the inspected region, several auxiliary conversions and long crossing edges separate the
main arithmetic operations. See [the Netron screenshot](netron-tinyllama.png).

XML/runtime inspection found 428 direct Constant-to-Convert branches. `HideConstantConverts`
summarizes only those single-input, single-output conversions at the downstream input ports.
The annotation preserves both the stored dtype and the effective output dtype/shape. It retains
the Constant ID as source provenance and the Convert ID as the effective value's identity. Two
different casts of one Constant must not appear to be the same converted value.

This is a display transformation, not constant folding of the model. It does not infer that a Convert
is an identity, read values, or bypass a runtime conversion whose input is not a visible Constant.
Longer conversion chains remain partly visible. Run it before ordinary constant hiding; in reverse
order the direct pattern is no longer visible, so the optional pass leaves those Converts unchanged.

The focused examples choose the first MatMul with a direct converted weight, and explicitly mark
omitted boundary connections. The example shrinks from five nodes/four edges to three nodes/two
edges while retaining the weight shape, storage type, effective type, source ID, and conversion ID.

## Rendering and the observed bottleneck

Graphviz HTML tables give every input/output a named attachment point. Text is escaped for both
HTML and Graphviz backslash substitution; XML-forbidden characters are replaced with U+FFFD.
Unknown operation types use neutral styling. Pass-produced border colors and constant annotations
are applied by the renderer without operation-specific transformation logic.

The initial full TinyLlama render with default `dot` optimization did not finish and was stopped.
Its long skip connections expand into many internal layout nodes. A diagnostic run reported 422,151
positioning nodes and 631,894 positioning edges, despite only 3,216 actual operations.

For more than 1,000 visible nodes, the renderer limits network-simplex/crossing optimization and uses
straight edges (`nslimit=2`, `nslimit1=2`, `mclimit=0.1`, `splines=line`). The hierarchical layout,
complete topology, port attachments, and labels are retained. Straight lines can cross unrelated
nodes, so the full SVG is an overview and the focused views are better for detailed inspection.
Smaller graphs keep Graphviz's default spline routing. These settings are based on the
[Graphviz layout attributes](https://graphviz.org/docs/layouts/dot/).

Rendering has a configurable timeout and writes to a temporary directory. An existing output is
replaced only after Graphviz succeeds. A timeout, missing executable, or malformed label produces a
clear error and cleans up temporary output.

The API and CLI also reject destinations that resolve to the input XML or associated weights before
loading the model. Existing hard links are checked by file identity as well as path spelling. The
statistics destination must also differ from the SVG, and statistics use an atomic replacement so a
failed write cannot leave a partially truncated file.

## Deliberate boundaries

Only tensor/data edges are drawn. Stateful ReadValue/Assign operations and their variable IDs are
present, but additional control-ordering relationships are not synthesized. Nested If/Loop bodies
remain inside their parent operation instead of being recursively expanded. Dynamic dimension bounds
are preserved, but equality relationships between framework-specific symbolic dimensions are not.

Possible next steps are selecting neighborhoods through the public CLI, collapsing repeated decoder
blocks with explicit boundary ports, and rendering hierarchy levels separately. None is needed to
extend the existing loader/pass interfaces.
