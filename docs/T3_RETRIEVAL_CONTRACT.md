# T3 scoped retrieval engineering contract

This implements the first provider-neutral prompt retrieval boundary in the
pinned T3 plan. It is the **single shared metric ablation**, using the maintained
NN-kNN `retrieve` path. No artifact is passed through NN-CDH or label reuse.
The query encoder is injected and versioned; the host explicitly supplies its
need, requested types and public task state. Learned encoders, specialized heads
and cross-head calibration remain separate required comparisons.

Cases retain original text, stable IDs, type, source, scope and lifecycle.
The bank is read-only and aligns active IDs only. Actual tensor/config hashes bind
the retrieval state; drift rejects old-version results. Torch module encoders
carry actual state hashes; opaque encoder callables are explicitly declared-only.
Read `T3_VERSION_BINDING.md` for verified failures,limitations and host regression.
Session admission requires expiration, persistent user admission requires
explicit retention intent, and domain/global admission requires validated
records. These flags are inputs from the caller's admission authority, not proof
that an approval process happened. The access object must come from trusted host
authentication rather than a retrieved artifact. This module never executes
tools, promotes memory or overrides quarantine. Private content is absent even
from the retrieval audit; only aggregate gated counts are disclosed.

Scope, expiration, quarantine, requested type and previously supplied IDs gate
the normalized selection. Each event returns a minimal original-evidence channel
and linked full audit with eligible IDs, distances, case biases, normalized
activations and the actual per-feature distance contributions from that same
calculation. Reliability remains explicit and uncertain cases stay uncertain.
Utility is unobserved until objective counterfactual feedback exists. The
conditional-credit helper uses removal loss differences and activation-weighted
C/H, never same-label correctness; it does not automatically update global state.

The host-directed loop bounds rounds, delivered cases, serialized evidence
characters and elapsed time at call boundaries. It stops on host readiness,
duplicate/near-duplicate textual needs, no novel eligible case or an exhausted
budget. Oversized evidence is withheld intact, and the audit explicitly says it
was not delivered. Providers must separately bound generation tokens/time;
character counts are not tokenizer counts or a compute guarantee. Only explicit
request/public-answer fields are retained; private reasoning is not requested.

Ten behavioral tests cover actual masked NN-kNN weights, feature-distance sums,
same-event channels, private-data exclusion, type/lifecycle/admission gates,
novelty, duplicate stopping, context withholding, public host completion,
cross-instance event uniqueness and signed counterfactual credit. The combined
T1/T3 suite passed 143 tests before the additional event-uniqueness test; no host
quality or generalization result follows from these engineering checks.

Next work: actual frozen-host requests and reuse, no-retrieval/standard one-shot/
NN-kNN prompt controls, causal interventions, task-conditional usefulness and
feedback admission/rollback, specialized metrics, internal/combined integration,
single-/multi-hop public benchmarks, biomedical evidence-answer evaluation and
bounded agent transfer. Participant and scientific success claims remain open.
