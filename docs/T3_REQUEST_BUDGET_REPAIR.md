# Uniform request budget repair

The first actual192-query collection retains a malformed request at position145:
the128-token output ends inside a JSON string. It continues logging every planned
question and rejects the incomplete manifest before fitting. This observation
motivates a new explicit treatment, not replacement of that question's request.
The original frozen run and all its outputs are retained.

Request collection and public evaluation now accept an explicit uniform request
token ceiling128/256/512, with128 unchanged by default. The next exploratory
treatment uses256 for every initial request, including successful questions.
It uses the same literal question-only prompt, frozen host and grammar, with
no individual retry or missing-query fallback. Answer/continuation ceilings stay
128. Public total ceilings become three calls/512 generated tokens at256; actual
opportunities and costs must still accompany results. This treatment differs
from the earlier128-token experiments and cannot inherit their cost claims.

Extended need manifests use version2 and bind the actual decoding ceiling and
total planned budget. The fitter rejects undeclared excess tokens or inconsistent
budget records. Actual-need public evaluation requires identical request decoding,
host files and format schemas/backend to the recorded training requests. The
standalone collection protocol also receives its measured format initialization
time, matching the final manifest header; old frozen headers stay preserved.

All59 T3 tests pass, including rejection of an undeclared129-token request,
acceptance under a declared256 budget, inconsistent budget rejection and immutable
separate answer/request settings. Completion under256 remains to be measured.
The old192 requests are not a successful actual-need training result.
