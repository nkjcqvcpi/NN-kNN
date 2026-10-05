# Direct public set utility preparation

The T3 plan§9.3 prefers direct conditional removal:
`u_i = J(q,S without i)-J(q,S)`. `model/t3/utility.py` now evaluates the exact
delivered full set,each single removal and empty. It preserves order/content and
never retrieves a replacement. Single-case empty is evaluated once. The sink
saves each outcome before the next evaluation; unexpected evaluator/sink errors
propagate without invented observations. Payload copies prevent a mutating
evaluator from altering subsequent interventions.

Named finite lower-is-better objective dimensions must remain fixed. Missing
or invalid outcomes require an explicit unobserved reason and generate no utility
or negative feedback. Per-case loss differences,set gain and two-case
complementarity are separate. Zero usefulness does not establish falsehood or
irrelevance. No global reliability label,parameter update or promotion follows.

`tools/t3_public_set_utility.py` consumes the complete audited80-trial constrained
public run and verifies its actual protocol,trials,journal and summary paths and
SHA256 hashes. It takes all16 questions×fixed/learned conditions:32 prospective
sets,including source failures. No valid-answer filtering is allowed. Missing
delivered sets remain explicitly unobserved. It binds the same9-file frozen
Qwen3-0.6B host,backend source,supplied original artifacts and dataset hashes.
Gold is used only after generation;the host receives the public question and
exact evidence subset. No hidden rationale,self-rating or dev training occurs.

Fresh full/subset calls use the same answer-stage prompt,greedy decoding,format
backend and128-token ceiling. Each set requires at most four calls,for a total
ceiling128 calls/16,384 generated tokens. This intervention compute is separate
from the source experiment. Full-set outcomes are newly generated;they are not
silently equated with the earlier generation or iterative trajectory. Primary
loss is1-answer F1; Hotpot support/joint losses remain separately named.
Invalid outputs count as task failures separately but leave case utility
unobserved. Actual NN activation exposure and objective C/H/Q are recorded
separately;missing comparable losses produce no credit and no automatic update.

All46 T3 tests pass,including complementarity,harmful-case sign,missing full
feedback,immutable subsets,single-case deduplication,objective drift and sink
failure before the next host. These tests are engineering evidence,not public
utility outcomes. The real32-set job is queued behind the current format/plain
GPU jobs;it requires completed strict plain results and audit before starting.
The driver stops on failed prerequisites. Independent objective/payload/credit
replay is required after generation. No public utility benefit is claimed yet.

Frozen144-file source
`1993a63f35dadedb4d4cf4d053940bed2184751ee6f573ed80c6032b6b309d40`,ZIP
`56e5982a772efa73892d43e01b2754f2647712a4e5319f8a70bef8bd91522047`.
Task evidence: `work/t3_public_set_utility_freeze.json`,
`work/run_t3_utility_after_strict_plain.ps1`, `work/t3_utility_driver.log`,
`work/t3_public_set_utility.log` and
`outputs/t3-public-set-utility-20261005/` (created only when the job starts).
Typed objectives/calibration,validated feedback batches,poisoning/rollback,
internal/combined and stronger-host controls remain required follow-up work.
