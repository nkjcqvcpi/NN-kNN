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
