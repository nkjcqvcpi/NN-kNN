# Optional frozen-host formatting contract

`model/t3/host_schema.py` defines request,answer and continuation schemas.
Requests must supply exactly readiness,need,requested types and observable public
state; answers must supply readiness,answer and ordered title/index citations.
No field is invented by the controller. Runtime checks reject wrong stage,
missing/extra keys,unsupported routes,negative or reversed citation indices and
unbounded strings. Validation is separate from truth,relevance and authorization.

The public pilot supports `--strict-schema` without a decoder change and optional
`--format-library` for a separately installed lm-format-enforcer0.11.3 backend.
The latter records backend source hash,declared and actual decoder schemas,
tokenizer preparation time and call stages. Existing plain behavior is the
default. Optional dependencies live in the task workspace,not the learner
environment. Host weights,prompts,candidate banks and token ceilings are retained.

Actual backend character-parser probing found that boolean `const` fails with
`AttributeError: 'bool' object has no attribute 'startswith'`. Decoder schemas
therefore use boolean type while runtime validation enforces readiness. The
declared contract is not mutated. Backend numeric minima and positional citation
types are also insufficient; post-generation checks remain authoritative.
Filtering admits JSON prefixes;128 generated tokens can still truncate a valid
prefix before completion. It is not a guarantee of JSON completion or semantic
truth. The third-party library is not patched.

Four focused schema tests and all42 T3 tests pass. Four actual character-parser
stage checks pass; wrong readiness is accepted by the weaker parser and rejected
by runtime validation. An actual80-trial frozen public-host run is underway and
already produces real constrained outputs,including truncation failures. No
success-rate or QA improvement is claimed until all trials and costs are audited.
The original plain baseline also differs in strict validation; run a strict
plain control before attributing outcomes specifically to token filtering.

Experiment freeze:141 files,source
`90aac35f106efb737679739d58f295e52fd66d44738c72b2e4e686d7b88f7d0e`,ZIP
`530f1eef49900ede6ceaa9b445429cb874330da99ace57b48e031683fddda77f`.
Task evidence: `work/t3_format_enforced_freeze.json`,
`outputs/neural-cbr-t3-format-parser-verification.json`,
`outputs/t3-public-host-20261005-format-enforced/` and
`work/t3_format_enforced.log`. Retrievals are journaled before subsequent host
calls and partial events retained on failures. The80-trial result is pending.

A completed31-trial prefix snapshot validates actual constrained generations:
55 calls,24 retrievals (18 NN/six BM25),1,152 manual candidate comparisons and six
schema failures. All24 events,including failed-trial partial events,match the
immediate journal exactly;no error trace gap remains in this prefix. Source/ZIP
bytes,case/model/encoder hashes,payload prefixes and objective scores replay.
The verifier makes no fresh LLM generations. This provisional31-trial boundary
is not terminal80-trial results. Evidence:
`outputs/t3-format-prefix-31/snapshot.json` and
`outputs/neural-cbr-t3-format-prefix-verification.json`.

The strict plain80-trial control is queued after the current GPU job. Its driver
waits for the verified process,requires the terminal80-trial summary and full
journal audit,then uses the identical141-file archive with `--strict-schema`
and no format library. It stops on failed prerequisites. Both runs receive full
journal audits;GPU trials do not overlap. Sequencing evidence:
`work/run_t3_strict_plain_after_format.ps1` and
`work/t3_format_then_plain_driver.log`. Runtime latency still includes changing
machine load and is not a hardware-isolated comparison.
