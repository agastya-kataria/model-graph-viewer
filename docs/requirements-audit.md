# Assignment requirements audit

Audited against the supplied [assignment PDF](../assignment.pdf) and the graph-readability follow-up
on 23 October 2026. This maps requirements to implementation and reproducible evidence; it does not
replace running checks on a new environment.

| Requirement | Implementation and evidence |
| --- | --- |
| 3.1: OpenVINO XML/BIN loading, separate adapter, metadata, meaningful failures | `OpenVINOLoader`; real-model parser tests include missing weights, invalid XML, disconnected parameters, and stateful sinks |
| 3.2: Independent operation IDs/names, attributes, visualization metadata | Validated graph dataclasses; duplicate-name tests; full original/view JSON snapshots |
| 3.2: Exact ports and multiple connections per node pair | Explicit port tables and port-aware edges; Split, parallel-connection, disconnected-input, and Assign tests |
| 3.2: Dynamic, partial, scalar, and unavailable shapes | Detached tensor metadata; loader/core tests distinguish all forms; JSON preserves them |
| 3.2: Valid graph modification | Indexed mutation methods; dangling endpoint, producer conflict, copy, invalid removal, and retained-port tests |
| 3.3: Configurable ordered passes independent of framework objects | `apply_passes`, configurable registry, ordering tests, documented custom pass |
| 3.3.1: Red border on every exact name match; others unchanged | `HighlightNodes`; duplicate-name, metadata-preservation, and parsed-SVG style tests |
| 3.3.2: Hide Constants with original port/type/shape and shared source IDs | `HideConstants`; single/shared/repeated consumer tests; mandatory annotations remain visible in compact mode |
| 3.3.2: Never copy/display values; preserve original model | Constant accessors prohibited in parser tests; detached snapshots; original XML/BIN preservation and output-alias tests |
| 3.3.3, optional: Inspect TinyLlama and demonstrate a useful improvement | [Netron screenshot](netron-tinyllama.png), converted-weight hiding, retained [before](../examples/svg/tinyllama.focus.before.svg)/[after](../examples/svg/tinyllama.focus.after.svg) detail |
| 3.4: Hierarchical directed SVG, relevant operation information, ports, styles, unknown operations | Graphviz renderer; XML/port/style/escaping tests; operation IDs/types/ports and required Constant annotations remain visible |
| 3.5: API/CLI, configurable passes, no-pass mode, errors, documentation | Public pipeline and console/module entrypoints; integration tests; [README](../README.md) |
| 4: Independent unit tests and real OpenVINO-to-SVG integration | Graph/pass tests plus parser, renderer, metadata, API, and subprocess CLI tests |
| 5: Efficient graph processing and approximately 5,000 nodes | O(V + E) construction/validation/passes; indexed connectivity; stress export retains 5,100 operation nodes |
| 5: Actual stage timings, graph sizes, environment, limitations | [Benchmark results](../benchmarks/results.json); README methodology and large-layout routing limitations |
| 8: Three OpenVINO model examples including TinyLlama | Included deterministic MLP/branching XML/BIN; pinned checksum-verified TinyLlama downloader |
| 8: Generated SVGs and reproduction | Three variants per real model, JSON sidecars, two detail demonstrations, synthetic stress output, scripts, lockfile, and README commands |
| Follow-up: Omit names and export omitted attributes | Compact labels by default; short scalar attributes remain visible; original/view JSON maps IDs to complete names/metadata |
| Follow-up: Reduce high-fan-out edge clutter while preserving connectivity | `ReferenceHighFanout`; node-level configurable threshold, exact output/input reference tokens, source links, use counts, unchanged logical graph edges |
| Follow-up: Investigate and demonstrate improvement | Preserved [previous TinyLlama SVG](../examples/svg/tinyllama.previous.svg); separate compact/reference variants; real fan-out [before](../examples/svg/tinyllama.fanout.focus.before.svg)/[after](../examples/svg/tinyllama.fanout.focus.after.svg) detail; geometry and stage measurements |

## Verification and acceptance

- The tests cover threshold boundaries, separate source output values, repeated consumer inputs,
  pass ordering, recomputation/idempotence, duplicate names, and unchanged logical connectivity.
- Every newly benchmarked SVG is independently checked against exact graph endpoint tuples, rendered
  reference IDs/links/text, source-use counts, original/view metadata, and tensor descriptions. Counts
  alone are insufficient once connections have two visual representations.
- `before_nodes`/`before_edges` and `after_nodes`/`after_edges` describe logical graphs. Drawn and
  referenced connections are reported separately, and together equal the transformed logical count.
- No-pass rendering preserves original topology. Compact label detail and JSON export are presentation
  defaults; Constant hiding and high-fan-out references require explicitly selected passes.
- Both detail demonstrations label their omitted external boundary connections. Their sidecars describe
  the selected neighborhood; full-model sidecars preserve the entire original and transformed graph.
- Installed-wheel verification is recorded in [verification.json](verification.json). Benchmark commands,
  actual environment/settings, sizes, geometry, and raw stage times are in the measured results.
- Fresh wheel installations run outside the source directory passed all 230 tests on each of Python
  3.11, 3.12, and 3.13. Dependency, source, script, and test hashes accompany that verification.

## Boundaries and submission

The follow-up explicitly permits omitting names/attributes and replacing drawn high-fan-out connections
with another representation that preserves connectivity. Exact port information, operation types,
highlighting, and required hidden-Constant annotations remain visible. Tensor values remain excluded.

Large graphs with references use polyline routing; unreduced large graphs retain a bounded straight-line
fallback. These comparisons record their actual layout policy. References do not impose layout rank
constraints, and full-model SVGs remain tall. Nested control-flow bodies and extra control-ordering
edges are not expanded. The README documents these limitations and potential extensions.

Section 8 requests a GitHub repository containing implementation, tests, at least three model examples,
generated SVGs, and complete reproduction/documentation. The [public repository](https://github.com/agastya-kataria/model-graph-viewer)
contains those materials; large TinyLlama weights are acquired through the pinned downloader.
The hiring email requests submission by Reply All, and the later reply confirms a public repository
is acceptable. This repository is the deliverable; email delivery is a separate final step.

The PDF also requires understanding and explaining the submitted implementation. The
[design walkthrough](design.md) covers identity versus names, runtime versus XML ports, Constant
sharing/conversion provenance, ordered in-place passes, reference connectivity, metadata snapshots,
and measured layout tradeoffs for that discussion.
