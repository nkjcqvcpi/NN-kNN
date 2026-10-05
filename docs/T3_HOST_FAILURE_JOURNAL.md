# Retrieval events survive later host failures

Public-host diagnosis found a missing audit path: `run_loop` held completed
retrievals in a local list,so a malformed later host decision raised before the
caller could retain their evidence/audit channels. Recording only successful
final answers hides retrievals that actually happened. Lost original event IDs
cannot be reconstructed or silently replaced by new replay IDs.

The optional `event_sink` now receives a deep copy of each completed retrieval
before the next host call. Delivery status is final at that boundary; empty or
over-budget payloads record `delivered_to_host=False`. Sink failures propagate
before proceeding. Normal host errors still propagate; no replacement need or
answer is invented. The public pilot writes `retrieval_events.jsonl` immediately
and retains `partial_events` on a later schema error. Successful final events
must equal the journal. This is single-run append logging,not distributed storage
or recovery from arbitrary disk failure.

`Budget.max_cases_per_round` also makes one-case-per-round iterative comparison
explicit while retaining the old default3. Total case/round/context ceilings
continue to apply. This permits actual one-then-novel-one retrieval rather than
calling a two-case one-shot result iterative.

34 focused retrieval/need tests pass,including actual same-core failure delivery,
withheld events and two novel one-case rounds. The213-test joint boundary passed
before the added failure-journal test. Four fresh actual NN core retrievals
replayed four observed archived real malformed host outputs. Delivered payloads
match the original answer prompts exactly; all four new linked evidence/audit
events survive the same parsing errors. These are new retrieval events,not fresh
LLM generations or recovery of original lost IDs. The source prefix used by this
replay is counted and hashed while the separate host experiment continues.

The public-host runner is available with explicit64-case SQuAD conditional pools
and Hotpot distractor contexts,prospective eight questions each,no retrieval,
BM25,fixed NN,loss-selected learned NN and iterative controls. It records actual
calls/tokens/time and resource ceilings; differing host/retrieval opportunity
counts must accompany outcomes. The first request prompt had missing-type schema
failures;27 completed trials/17 schema failures/31 recorded completed host calls
were preserved before diagnostic termination. In-flight generation cost is not
claimed by that count. Stage-separated request prompt v2 proceeds separately;
the completed public outcome analysis is still pending. Future runs use the
journal; old errored trials without events remain an explicit evidence gap.

Source `cc396e87b62a1ab38d9eb5d05c7472fc7acc825f8a1dfd778d0adf8cbb2eb1b0`,
139 files,ZIP SHA256
`e08a42f47603e877cac9c010790ef0aeb34c1096ce10c1a6a5a5d548cac5cbf1`.
Source/fullZIP bytes validate. Task evidence:
`outputs/neural-cbr-t3-failure-journal-verification.json`,
`outputs/t3-public-host-20261005/termination_diagnosis.json` and distinct source
archives/logs for request v1,v2 and the journal implementation. No public task
benefit,utility feedback,typed-head comparison or full T3 completion is claimed.
