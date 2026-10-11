# Assignment requirements audit

Audited against the supplied [assignment PDF](../assignment.pdf) on 11–12 October 2026.
This maps the brief to implementation and evidence; it is not a guarantee against every possible bug.

| PDF section | Requirement | Implementation and evidence |
| --- | --- | --- |
| 3.1 | OpenVINO XML/BIN loading, metadata extraction, separate adapter, meaningful failures | `OpenVINOLoader`; real-model tests in `test_openvino_loader.py`, including missing weights, invalid XML, and stateful sinks |
| 3.2 | Independent node IDs/names, attributes, visualization metadata | `Node`, `Visualization`, graph validation, duplicate-name tests |
| 3.2 | Exact ports, multiple connections per node pair, model boundaries | Explicit node port tables and port-aware `Edge`; Split, parallel-connection, disconnected-parameter, and Assign tests |
| 3.2 | Dynamic/partial/scalar shapes and missing metadata | Portable shape/dtype descriptions; bounded and unbounded dimensions, dynamic rank, scalar, and unavailable-metadata tests |
| 3.2 | Valid graph modification | Indexed mutation methods; tests for dangling endpoints, producer conflicts, invalid removals, copies, and retained ports |
| 3.3 | Ordered, configurable passes independent of framework objects | `apply_passes`, pass registry, exact order tests, custom pass example in README |
| 3.3.1 | Red border on every exact name match; other nodes unchanged | `HighlightNodes`, duplicate-name and unchanged-metadata tests, parsed SVG style checks |
| 3.3.2 | Hide Constants with port/type/shape annotations and shared IDs | `HideConstants`; single/shared consumer tests, repeated inputs, original sharing provenance, unrelated edges, idempotence |
| 3.3.2 | Never copy/display values or modify the original model | Constant value accessors are prohibited in parser tests; detached graph data, source-file preservation tests, guarded output paths |
| 3.3.3, optional | Inspect TinyLlama in Netron and demonstrate a useful pass | [Netron screenshot](netron-tinyllama.png), `HideConstantConverts`, [focused before](../examples/svg/tinyllama.focus.before.svg)/[after](../examples/svg/tinyllama.focus.after.svg), explanation in [design walkthrough](design.md) |
| 3.4 | Valid hierarchical SVG, directed edges, exact ports, operation information, styles, unknown operations | Graphviz `dot` renderer; XML/port/style/escaping tests and all nine committed SVGs |
| 3.5 | Python API and CLI with input/output selection, no-pass mode, ordered pass options, errors, documentation | Public API, console/module entrypoints, CLI integration tests, [README](../README.md) |
| 4 | Independent unit tests and a real OpenVINO-to-SVG end-to-end test | Graph/pass unit suites plus real-model parser, renderer, API, and subprocess CLI tests |
| 5 | Efficient processing and support for about 5,000 nodes or more | Adjacency/producer indexes, in-place passes; stress artifact contains 5,100 rendered nodes and 5,050 connections |
| 5 | Actual stage timings, sizes, environment, limitations | [Raw benchmark results](../benchmarks/results.json), README table, documented Graphviz bottleneck and layout tradeoff |
| 8 | At least three OpenVINO examples including TinyLlama | Deterministic MLP and branching XML/BIN examples; pinned, checksum-verified TinyLlama acquisition script |
| 8 | SVG examples and reproducibility materials | Nine SVGs; model generator, download script, focused example, benchmark script, dependency lock, clean-wheel evidence |

## Verification performed

- The ordinary suite covers each component and the complete public API/CLI pipeline. Clean installed
  wheels are tested outside the source directory; [verification.json](verification.json) records
  exact Python versions, outcomes, package origins, and source/test/dependency hashes.
- A fresh local Git clone resolved every documentation link and reproduced all four small model files
  byte-for-byte. Both small models completed CLI transformation/rendering from outside the clone.
- Each full-model SVG was compared against the current parsed/transformed graph: node identities and
  every edge's source node, output port, destination node, and input port agreed. Counts and file sizes
  also agree with the recorded benchmark evidence.
- Both downloaded TinyLlama files matched their pinned sizes and SHA-256 digests. The graph contains
  the 428 direct Constant-to-Convert patterns described in the optional-pass explanation.
- Additional temporary review probes used seed `902103`: 3,000 valid synthetic graphs matched an
  independent Constant-hiding reference, and 2,000 Constant/Convert graphs passed checks for sharing,
  provenance, ordering, and idempotence. These probes supplement, rather than replace, the committed
  regression suite.

## Boundaries and submission

The implementation's limitations are explicit: nested control-flow bodies are not expanded; extra
control-ordering edges are not synthesized; longer conversion chains are not collapsed; and large
graphs use straight edges that can cross nodes. Full topology and port information are retained.

The requested deliverable has been prepared as a **local Git repository**. The PDF ultimately asks
for a GitHub repository; pushing it and submitting that URL remain the publication step, deferred
by the selected local-only delivery scope.

The PDF also requires understanding the submitted code. The [design walkthrough](design.md) explains
operation identity versus names, runtime versus XML port indices, Constant sharing, in-place graph
transformation, converted-value provenance, and the observed rendering bottleneck for that discussion.
