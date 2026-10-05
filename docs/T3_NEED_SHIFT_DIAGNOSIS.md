# Public host need input shift diagnosis

The frozen-host request instruction asks for the public question,but actual
requests often append public instruction words,such as “Please provide evidence”.
Literal questions differ in64/64 requests;the declared lexical token sequence
matches in20/64. Actual hash vectors differ in44/64. The controller keeps those
model-issued needs;no hidden planner or replacement question is introduced.

A separate diagnostic deliberately supplies the original public question to the
same frozen bank/metric/encoder under the same type/case budget. These are NEW
explicit query counterfactuals,not host-issued requests or reconstructed original
events.48 fresh actual NN retrievals and16 BM25 query controls compare all64
source retrievals,including failed-answer trials. Model/bank hashes remain equal;
1,776 independently calculated feature contributions/distances pass.

| Condition | Comparisons | Changed hash vectors | Changed ordering | Changed membership |
|---|---:|---:|---:|---:|
| BM25 | 16 | 11 | 3 | 2 |
| Fixed NN | 16 | 11 | 4 | 4 |
| Learned NN | 16 | 11 | 5 | 4 |
| Iterative | 16 | 11 | 2 | 2 |
| Total | 64 | 44 | 14 | 12 |

The need-matching fitter currently trains on public question text,while actual
evaluation receives model-generated needs. Additional instruction tokens change
the lexical representation and sometimes selected cases. This establishes an
input-shift mechanism;it does not establish a QA benefit from overriding the host
or explain all weak results. The earlier tune recall degradation exists even
without this shift. Original primary requests,answers and audit IDs are untouched.
No dev answers/support annotations are used by this diagnostic or for training.

Next fit/calibrate on actually emitted train-only needs or improve the declared
need representation under prospectively fixed controls. Keep original-query
retrieval as an explicit control rather than silently cleaning/replacing host
requests. Direct set utility is evaluated separately on the actual delivered
sets. Stronger-host/typed-head/internal comparisons remain open.

Task evidence: `work/diagnose_t3_need_shift.py` and
`outputs/neural-cbr-t3-need-shift-diagnosis.json`. Source141-file archive
`90aac35f106efb737679739d58f295e52fd66d44738c72b2e4e686d7b88f7d0e`,the completed
80-trial constrained input and its full journal verification are bound.
The verifier performs actual core retrievals but no new LLM generations.
