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

## Fresh actual diagnostic and full run

Both real bounded-profile requests complete:118/186 output tokens,512/510 need
characters,182 total input/304 output tokens and63.77 generation seconds. Exact
tokenizer/source-prompt/host/decode/schema/output/journal checks pass. The needs
retain the original repeated prefixes; format completion has not repaired their
semantic quality. The token grammar can close before512 when the next whole token
would exceed the bound. An initial verifier incorrectly required exactly512 in
both cases; inspection of the actual510-character second output corrected that
assumption. No output was changed.

The bounded192-query collection has actually started,followed by six matched
20-epoch fits,six independent full retrainings and common-query evaluation. Two
public exploratory32-trial arms then compare question-trained and actual-need
metrics using seed8,identical bounded grammar/request256/answer128 conditions.
Seed8 is recorded before these new public outputs; earlier exploratory results
remain known. Both arms use the existing16 exploratory questions;reserved64
remain untouched. This is redesign evidence,not independent confirmation.

Frozen149-file source:
`e26aac9c13ab3e00e7c022c4ad02d6b0469f034735be0c0e18975665e19a3dd7`,
ZIP `7a87a29675d5190345bd1ad5d8556cc5fdbf8e02ce7b1ebe7f4b9dba903a8963`.
Task evidence: `outputs/t3-bounded-request-diagnostic-20261006/`,
`outputs/neural-cbr-t3-bounded-request-probe-verification.json`,
`work/t3_bounded_request_plan.json`,`work/run_t3_bounded_request_256.ps1`,
`work/t3_bounded_request_256_driver.log`. The first sequencing preflight referred
to the new output rather than the preserved old failure;it stopped before any
generation. The corrected driver verified no new output existed and preserved
that preflight log separately before starting. Session90847 is the live full run.
