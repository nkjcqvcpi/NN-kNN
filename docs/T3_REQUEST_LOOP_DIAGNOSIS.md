# Actual request repetition diagnosis and bounded grammar

Both complete192-query treatments fail on the same two train-only questions,
positions145/174. Under128 and256 tokens respectively, these outputs repeat
question suffixes indefinitely inside the need string until truncation. The
other190 actual output strings are exactly identical between treatments.
Increasing the token budget therefore did not repair collection.

128:192 calls/190 valid/two failures/16,705 input/9,878 output tokens,
1,974.32 generation seconds.256:192/190/two/16,705/10,134,1,874.13 seconds.
Both exact complete journal/parse/error/accounting audits pass and verify that
incomplete coverage is rejected. Neither actual-need fitting nor its public
comparison starts. Timings are observed totals,not hardware-isolated speed claims.

An optional bounded request profile now limits need strings to512 characters and
observable state to128. The same profile applies to initial and continuation
requests; answer schemas are unchanged. It changes the permitted output grammar,
not the request prompt or an emitted need's content. Runtime and manifest loading
also validate these limits. Nonstandard profiles require an explicit version2
treatment. Default standard schemas remain available for historical controls.
Training/evaluation schema bindings require the identical grammar profile.

Character-by-character actual backend replay of both failed256-token outputs
rejects further need-string extension at character543,where the512-character
need limit is reached. This proves the grammar blocks the observed continuation,
not that a model will subsequently emit a valid complete request or a useful
need. A fresh two-question diagnostic precedes full collection; its selection
is failure diagnosis,not a valid fit/evaluation sample. Full192 coverage is still
required before any fitting. Repeated content retained within the limit can
still be poor need matching and must be measured rather than silently cleaned.

All61 T3 tests pass. The previous uniform-budget revision passed all235 repository
tests with one existing T1 tensor-to-scalar warning. Task evidence:
`outputs/neural-cbr-t3-failed-need-collection-verification.json`,
`outputs/neural-cbr-t3-failed-need-collection-256-verification.json`,
`outputs/neural-cbr-t3-request-loop-diagnosis.json`.
The original147 and uniform256149 source archives remain unchanged.
