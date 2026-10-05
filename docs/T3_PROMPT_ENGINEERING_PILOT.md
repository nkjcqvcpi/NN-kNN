# Frozen-host request and original-evidence reuse pilot

The actual local Qwen3-0.6B host, pinned at
`c1899de289a04d12100db370d81485cdf75e47ca`, now emits explicit JSON retrieval
requests, receives original scoped artifacts through NN-kNN, and emits a public
answer. All 596,049,920 host parameters remain frozen. The shared retriever is
also frozen: a declared 64-dimensional SHA256 lexical count representation,
NN-kNN geometry and fixed initial biases, with zero retrieval training steps.
This is a constructed four-fictional-entity engineering fixture, not semantic
retriever training or a public T3 downstream benchmark.

The successful source is
`076378f506ba435f552c8d45e30f79a297897519e4e5d10c8af3d2c93bd20d37`,
112 files, ZIP SHA256
`a4054aa4367815133e34dbf60bcc3f061f184079c22cdfd1137234b129f60590`.
The source contains unrelated in-progress regression work; this pilot invokes
only the archived T3 driver/core. Every local model/tokenizer file passes the
pinned manifest hash and byte checks; no remote code or model download runs.

| Condition | Exact color answer | UNKNOWN abstention | Actual retrieval rounds |
|---|---:|---:|---:|
| No retrieval | 0/4 | 4/4 | 0 |
| NN-kNN one-shot budget | 4/4 | 0/4 | 1 per task |
| NN-kNN iterative budget | 4/4 | 0/4 | 1 per task |
| Target case quarantined | 0/4 | 4/4 | 1 per task |

No-retrieval abstention is appropriate for unknown fictional facts, not evidence
of general host failure. The quarantined condition excludes the target before
ranking and the host abstains despite other-entity evidence. The iterative
condition permits two cases/rounds, but all tasks finish after one retrieval.
It demonstrates readiness stopping, not a multi-hop or iterative advantage.
One-shot permits one case; context budgets differ explicitly. No-retrieval and
retrieval/answer-stage prompts also differ, so this cannot establish matched
scientific RAG superiority. Quarantine reranks/refills the candidate set; it is
not direct conditional utility `J(S without i)-J(S)` without replacement.

There are 16 trials, 28 actual host calls, 3,740 input and 788 output tokens.
Prompts, outputs, decoding settings, case channels, token counts and timings
are logged. An independent reconstruction safely loads the retriever and
replays all 12 actual retrieval events: eligible IDs, original evidence,
distances, biases, activations and feature contributions match. Event IDs are
unique and linked; actual frozen retrieval tensors remain unchanged. Stored
host JSON/public answers and exact outcomes are checked; the replay does not
claim another 28 independent host generations.

Two rejected prompt versions remain preserved. The first copied the example
need literally, omitting the entity: one-shot 1/4 and iterative 2/4, with wrong
answers after quarantine. The second produced entity-containing requests but
continued requesting after receiving evidence, exhausting budgets. The working
version separates request-stage and evidence-answer instructions. The loop now
rejects missing/nonboolean readiness rather than inferring it from another
field; eleven contract tests pass. With the in-progress five regression tests,
the current combined T1/T3 suite passes 150 tests.

Required next work: exact-set causal removals, matched standard semantic RAG/
no-retrieval/prompt controls on objective public single-hop and multi-hop tasks,
trained/type-specific/calibrated retrieval, feedback promotion/rollback,
internal and combined integration, stronger/cloud/API hosts, biomedical evidence
and answer evaluation, and bounded agent transfer. No human, biomedical,
competitive-quality or scientific readiness result has been established.

## Exact-set removal extension

`tools/t3_prompt_pilot.py --direct-set-removal` now adds a separate causal audit
after the sixteen trials. It deletes each displayed one-shot artifact from the
already selected evidence set, does not refill or retrieve again, and calls the
same frozen host at the answer stage. Audit verifies the full rendered prompt
is byte-identical except for that evidence JSON replacement. It links the
source event, case ID, full/removed sets, public outputs, objective0/1 loss and
the actual retrieval activation used in conditional C/H/Q credit.

Frozen source `b1676e208a0f5346b364f609f03b50d95bffce8b73a662b0fdc4165d8b27abca`,
113 files, ZIP SHA256
`e2516d89a3ee3545e2e1ef083979b8e6d75eb45995b41a80d2922466f0b7fb89`.
The repeated sixteen outcomes match the previous fixture. Four additional
actual generations change the correct color to UNKNOWN after removing its
sole supporting record. Each has loss0→1, activation1, conditional utility1,
C1/H0 and smoothed Q2/3. Credit stays a conditional engineering observation;
there is no global reliability/parameter promotion. Single-record sets do not
test redundancy, complementary support, poisoning or rare-capability recovery.

There are32 total calls,4,160 input and852 output tokens in the extended batch.
An independent audit checks all four exact-set operations, prompt identity,
actual stored responses, objective loss, event weights and C/H/Q calculations.
Quarantine/reranking remains a separate intervention. Public benchmarks and
matched contemporary controls remain required.
