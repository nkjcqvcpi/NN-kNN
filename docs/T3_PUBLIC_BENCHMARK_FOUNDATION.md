# Public QA data and scoring foundation

`model/t3/benchmarks.py` reads explicit SHA256-checked SQuAD1.1 and HotpotQA
sources. PublicTask contains question,immutable original cases and grouping;
`host_task()` exposes only question and ID. Gold answers/support annotations
remain separate. Passage JSON preserves original title/text or title/sentence
lists without answer marks or supporting-fact flags. Stable nonnegative63-bit
IDs bind source and content; collisions and duplicate question IDs reject.

This is data/scorer preparation,not a retrieval or host result. SQuAD passages
require a declared retrieval corpus before comparison. Hotpot uses its supplied
distractor context,not full Wikipedia. No Hotpot train/fullwiki/test download is
claimed. Public dev annotations cannot train or select models in this package.

Official sources: [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) and
[HotpotQA](https://hotpotqa.github.io/),both declaring CC BY-SA4.0 for data.
SQuAD files are pinned to publisher repo
`eee5fdbf62f8613a7812b03419e6b29617b74fd1`.
CMU Hotpot HTTPS and HTTP timed out; the alternative is explicitly the
`hotpotqa/hotpot_qa` organisation mirror at
`1908d6afbbead072334abe2965f91bd2709910ab`. Its parquet hash and converted JSON
hash are preserved; byte identity with the unavailable CMU JSON is unverified.
Full7,405-row comparison verifies all original fields and text survive conversion
and equal-length column lists prevent silent zip truncation. Optional PyArrow
25.0.1 is isolated in the task workspace; maintained learner dependencies unchanged.

| Artifact | Questions | Unique source-bound paragraph cases |
|---|---:|---:|
| SQuAD1.1 train | 87,599 | 18,896 |
| SQuAD1.1 dev | 10,570 | 2,067 |
| HotpotQA distractor validation mirror | 7,405 | 73,700 |

Train SQuAD fit/tune partition uses article-title hashes before question sampling;
the two groups share no article. Prospective128 fit/64 tune IDs are selected by
public question-ID hashes. Each public dev split has8 exploratory/64 reserved
IDs selected before any host runs,never by answer,support,length or measured
success. These are engineering samples; budgets/prompts/confirmation protocol
remain to be fixed. Public dev data are not a hidden official test set.

The mirror contains one unavailable support annotation:
question `5ae61bfd5542992663a4f261`,title `Jimmy Butler (basketball)`,sentence902
in a five-sentence paragraph. Strict loading rejects it by default. Explicit
`allow_unaligned_support=True` preserves the entire question and original gold
fact while recording `unavailable_supporting_facts`; no correction or deletion.
Its evidence attainability is limited,and its contribution must remain visible
in any evaluation. The initial strict-audit failure is preserved.

511,156 real-annotation scoring comparisons agree with separately hash-pinned
publisher evaluators: all SQuAD examples under four answer probes and all Hotpot
examples under four answer×four support probes. Independent implementation
checks EM,alias/multiset F1,yes/no/noanswer rules,support/joint metrics. SQuAD1.1
uses the current pinned **v2 answerable** scorer's EM/F1 conventions,including
empty-normalized answers; it is not claimed to import a missing v1.1 script.
Hotpot uses its official v1 script at publisher commit `36358534`.
Evaluator hashes are required before importing their code; a data manifest alone
cannot substitute executable evaluator code. Wrong hashes/unaligned spans,
original preservation,annotation separation and metric edge cases have focused
tests.33 T3 tests pass; previous joint boundary205 passed before these four tests.

Task evidence: `work/t3-public-data-20261005/manifest_mirror_complete.json`,
`outputs/t3-public-data-audit-20261005-preserved-gaps/verification.json`,
`outputs/neural-cbr-t3-public-conversion-verification.json`; failed downloads,
missing v1.1 evaluator and initial strict annotation checks remain in task logs.
Source `9ea49b9e196d7837eaf18d1213e50aa8968d5dcee39f28179348d22888967c41`,
135 files; ZIP SHA256
`91f40cb40d13e0eaf21d485875ed397cf5476044f413ead936a138b843d2411f`.
Full source ZIP/bytes and all downloaded payload hashes validate.

Next: freeze corpus/context/host/prompt/compute protocols and run no retrieval,
standard lexical one-shot,NN-kNN one-shot and iterative controls; distinguish
fixed lexical geometry from actually learned need compatibility and validated
utility learning. Add Hotpot training data before any Hotpot fitting. Typed heads,
calibration,feedback and internal/combined comparisons remain required.
