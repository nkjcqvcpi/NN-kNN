# Candidate lexical term-frequency robustness

The repaired Northwestern need retains repeated suffixes and causes a fixed-NN
source miss while BM25 still ranks its source first. The public need is not
silently replaced. Optional sublinear (1+log word count) and binary word-frequency
features now compress repeated words before hashing,with the same L2 normalization
and256 dimensions. These are lexical robustness ablations,not semantic encoders
or an empirical declaration that compression improves retrieval.

Case text and query text use the identical declared encoding. The training loss,
case builder,query module and public runner carry the same mode;training/retrieval
reject a mismatched case/query mode. Checkpoint provenance supplies the encoding
to public and internal drivers. Nondefault modes have distinct encoder metadata
and case configuration. Hash collisions are handled after per-word compression.
The original count loop and default encoder description are bit-exact controls.

All71 T3 tests pass:independent per-word/hashed/L2 formulas,small-dimension collision
checks,legacy default equality,binary repetition invariance,training/retrieval
geometry equality and mismatch rejection. Real default-training state comparison,
matched actual-need fits and independent manual-gradient/state/common-query
replays remain required. No new host requests are needed;the audited192 actual
needs and fixed128/64 split can be reused. Public query generation and budgets
must retain the identical bounded treatment when evaluating new checkpoints.

Existing bounded149 and internal152 jobs keep their frozen sources. The current
live default-public results do not evaluate these new encodings. Source matching
is still distinct from outcome utility;rare words/cases are not labeled harmful
because a repetition-robust encoding changes their ranks.
