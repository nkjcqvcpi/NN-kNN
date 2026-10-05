# T3 immutable provenance and actual state binding

Version labels previously checked only case membership. Actual old/new archived
core probes show the previous implementation accepted `validated="false"`,allowed
metadata replacement without a new bank version,returned changed-bias geometry
under the original model version,and rejected preallocated banks with metadata
for active cases only. These four counterexamples now behave correctly.

Case admission requires actual boolean flags,nonnegative integer IDs,nonempty
original content/provenance,typed owner and finite nonboolean expiry. The bank
holds a read-only copy of frozen Case records. Request/access collections are
copied into immutable tuples/frozensets. Alignment and eligibility use active
stable IDs; inactive capacity rows do not need metadata or count as gated cases.

Every event binds the declared model version to SHA256 of actual module topology,
tensor state,core configuration,active count and effective retrieval attributes.
Verify before encoding,after encoding,after retrieval restoration and before
return. Drift rejects the result; reconstruct a new retriever for a new model or
bank. This is detection at boundaries,not isolation from arbitrary concurrent
external model writers. Bind source code separately using the source archive.

For an injected Torch module encoder,hash its topology/tensor state and audit
`encoder_verification="module_state"`. Opaque callables explicitly report
`declared_only` and no encoder-state hash: a caller's version string cannot verify
closure state,external resources or arbitrary non-tensor behavior. Module hashes
also do not prove arbitrary external/non-tensor behavior is frozen.

Retrieval clears the transient feature cache so previously populated features
cannot override the bound model. Restore its old cache,the exact config mapping
and every submodule training mode,including mixed train/eval states. Module query
encoders run in eval mode with all prior modes restored. Original evidence and
human audits still derive from one actual normalized masked core computation.
No automatic memory promotion,tool execution or learned metric is introduced.

29 T3 contract tests cover these failures and lifecycle behavior. The full
T1/T2/T3 suite passes205 tests (one pre-existing tensor-to-scalar warning).
Actual frozen Qwen3-0.6B regression:16 fictional trials/32 host calls,12 retrieval
events and four no-refill direct-set removals. One-shot/iterative each4/4 correct,
no retrieval0/4,quarantined target4/4 UNKNOWN. Fresh core replay verifies all
evidence,geometry,provenance and actual state hashes; host counterfactual prompts
and credit replay exactly.4,160 input/852 output tokens; retrieval tensors unchanged.
This preserves earlier behavior; it establishes no public benchmark benefit or
multi-hop advantage.

Source `2558358dbece90ace1155e9989bfccc1e16535962f4c84351dad7ef22792e45a`,
132 files; ZIP SHA256
`293afe38f9be4234ffe92dca5253d5ff69b0b048687b3f559faa2eb56c034f45`.
Task artifacts: `work/t3_version_binding_source_archive/`,
`outputs/t3-version-binding-host-20261005/`,
`outputs/neural-cbr-t3-version-binding-probes.json`,
`outputs/neural-cbr-t3-version-{host,removal}-verification.json`.
Career-2027 HEAD was freshly checked as `cd772776` on October5; imported18
authority documents remain unchanged. Public matched benchmarks,learned metric,
specialized heads,feedback and internal/combined integration remain open.
