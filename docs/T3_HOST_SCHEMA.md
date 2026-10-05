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
