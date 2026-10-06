# Candidate frozen-host internal output interface

T3§4 requires a declared internal interface on the same open-weight checkpoint,
with stable case lineage and trainable scope. This implementation is one candidate
for empirical comparison,not the settled architecture or arbitrary host fine-tuning.
It mixes next-token probabilities with an activation-weighted histogram of original
retrieved evidence tokens. Public QA evidence uses title plus passage/sentence text.
Activation is need match,not outcome utility. Host,retrieval and interface parameters
remain frozen;the gate is a fixed supplied alpha,not tuned on public outcomes.

Every histogram retains original stable IDs,provenance,content hashes,activation
weights,token counts and tokenizer version. Control/special tokens are excluded.
The mixture renormalizes artifact mass over the host's already-permitted tokens,
preserves hard grammar masks and leaves original logits exactly unchanged when
alpha is zero or no artifact token is permitted. It rejects cross-task batches,
misaligned IDs,invalid activations or an empty permitted host distribution.
This is a lexical output interface;it loses evidence order and is not a semantic
hidden-state fusion model. Internal-only citation grounding may consequently fail.
It does not authorize tools or infer evidence reliability from match scores.

`tools/t3_internal_host_pilot.py` prepares64 actual answer calls:16 existing
exploratory questions × none/prompt/internal/combined. All conditions share the
same actual archived host-produced needs and a fresh NN-kNN retrieval of the same
original conditional sets. The initial source request is not regenerated or
invented by the driver. This is a matched mechanism experiment,not a fresh full
retrieval-loop comparison. Source answer failures remain in the denominator.
Reserved questions remain unused and all annotations are used only for scoring.

All conditions use the same9-file Qwen3-0.6B,greedy128-token answer ceiling and
strict answer grammar. Prompt/combined receive original evidence in the prompt;
internal-only receives its token distribution through the declared output
interface with no evidence payload in the prompt. None receives neither. A fixed
alpha0.1 is declared before the new answers. The shared original request cost,
shared fresh retrieval,interface tokenization and actual answer costs remain
separate;resource ceilings do not establish equal realized compute. Completed
malformed outputs remain observed failures and are journaled before parsing.

Four tests independently verify the probability formula,stable lineage,exact
zero/no-mass controls,hard-mask preservation,special-token exclusion and invalid
cross-task/activation rejection. All65 T3 tests pass. Actual host generation and
independent case/token/score replay remain required before any benefit claim.
The current bounded192-query experiment retains its frozen149-file source.

A separate public-host accounting fix uses its declared total generated-token
ceiling instead of the historical literal384,which would wrongly reject a
legitimate request256 plus two128 continuations. Its request/answer ceilings
remain unchanged. Existing frozen jobs are preserved;their outputs must be
interpreted against the declared and actual budgets independently.

## Real-case preflight and probability trace repair

Sixteen actual fresh NN-kNN retrievals exactly reproduce the recorded source
sets/state/bank bindings. Independent geometry verifies592 candidate distances
and feature contributions. The actual Qwen tokenizer produces32 original-case
histograms;32 full/masked synthetic-logit probability checks pass. This is real
case/geometry/tokenizer evidence,not a fresh host answer result.

The initial151,936-token float32 softmax recomputation differed by up to6.03e-7
in the first preflight;float64 normalization and direct exponentiation of the
composed log probabilities agree with the independent mixture to roughly1e-9.
A separate large-vocabulary regression reproduced a2.89e-6 error in the stored
chosen-token diagnostic. The trace now records the composed probability directly
instead of performing a second large float32 softmax. The old failure logs remain.
Greedy returned logits are unchanged by this trace correction. All66 T3 tests
pass,including the large-vocabulary regression.

Each internal step now records permitted artifact token IDs,actual permitted
mass,chosen host/mixture probabilities and hard-mask preservation. This permits
independent reconstruction from original-case token counts and activations,with
actual generated token IDs tied to the recorded greedy selections. Interface
preparation time is reported separately from generation. Task evidence:
`outputs/neural-cbr-t3-internal-preflight-verification.json`,
`work/t3_internal_preflight_float32_failure.log`,
`work/t3_internal_large_before_fix.log`,
`work/audit_t3_internal_host.py`. A fresh64-call host run and its full independent
audit remain necessary. No internal quality benefit or empirical architecture
selection is established by this preflight.

## Queued actual experiment

The four-arm64-call experiment is now queued as session96676 behind the verified
bounded experiment owner23992. It is independent of that experiment's scientific
outcome and waits for its GPU use to finish. Source152 files:
`225483f26fb77911321ac86ac52db90095e5b88d8d0a14500fe237e2ab8e8a40`,
ZIP `54a5c8265b13ab9aafd6196458c437ec675875b048abf1c6749718ee0065110a`.
The prospective plan records alpha0.1 and all four arms before new answers.
The independent auditor checks64 calls,16 fresh original-query/core retrievals,
592 manual candidate contributions,actual tokenizer counts,original case
provenance,per-step permitted artifact mass and selected-token mixture formulas,
generated IDs,strict failure scoring and complete journals. It does not perform
fresh independent LLM regeneration or certify the internal architecture.

Task paths: `work/t3_internal_experiment_plan.json`,
`work/run_t3_internal_after_bounded.ps1`,`work/t3_internal_after_bounded_driver.log`.
Future results: `outputs/t3-internal-host-20261006/`,
`outputs/neural-cbr-t3-internal-host-verification.json`.
All242 repository tests pass with the existing T1 tensor-to-scalar warning.
The career-2027 remote HEAD was rechecked onOctober6 and remains the imported
`cd77277600841438d98ac5793867da28b8ec6dd0`. Broader semantic/typed integration,
competitive outcomes,poisoning/rollback,scope calibration,stronger/cloud hosts
and qualified/human validation remain open.
