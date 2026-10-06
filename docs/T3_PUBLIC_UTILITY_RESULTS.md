# Direct public case-set utility results

All32 actual fixed/learned sets complete:128 fresh model calls,52,730 input/3,532 output tokens,743.30s generation time;four invalid outcomes remain observed unit-loss task failures. All64 removal comparisons are observed.

| Dataset/condition | Helpful/harmful/zero pairs | Full F1 | Empty F1 | Mean marginal | Mean full-set gain | Mean pair complementarity |
|---|---:|---:|---:|---:|---:|---:|
| squad/fixed | 1/6/9 | 0.1538 | 0.2215 | -0.1476 | -0.0676 | -0.2275 |
| squad/learned | 3/5/8 | 0.1829 | 0.2215 | -0.0223 | -0.0386 | -0.0061 |
| hotpot/fixed | 4/0/12 | 0.2232 | 0.0000 | +0.1295 | +0.2232 | +0.0357 |
| hotpot/learned | 4/0/12 | 0.2500 | 0.0000 | +0.1562 | +0.2500 | +0.0625 |

Total64 conditional pairs:helpful12,harmful11,zero41. Positive means removing the case worsens the declared task loss;negative means removing it improves this host/query/set outcome. Zero does not establish falsehood or irrelevance.
All32 fresh full-set outputs exactly equal their archived source outputs. Every subset prompt differs only by its recorded evidence removal;no refill occurs.128 call/outcome journals,objective scores,activation exposure,C/H/Q and complementarity pass independent recorded-output/manual-arithmetic replay.

all32 prespecified engineering sets,including source failures; direct actual full/remove/empty counterfactuals; no replacement or global truth/promotion claim; near-zero tolerance1e-12 is reporting only

Evidence:outputs/t3-public-set-utility-20261005/;outputs/neural-cbr-t3-public-utility-verification.json;outputs/neural-cbr-t3-public-utility-results.json.
Source144 files:aaba914ea1445c77bcb474a0cb0f18578460a372905f02cafe9222746afee797;ZIP62defc4cca7082cb471fd3741cdf3e43a22b8261f7d87bc8f638f7e2bbb3bb2c.

SQuAD full-set gain is negative in both groups. The model can perform worse with
the actual retrieved artifacts than with empty evidence under the same answer
instructions. Some harmful differences include formatting failure;none implies
the original paragraph is false. Need matching cannot substitute for utility.
Hotpot full-set gains are positive on these eight questions and a few two-case
sets show positive complementarity,but support/joint losses remain unchanged:
the model still does not supply correct supporting facts. Answer improvement
alone is insufficient for the plan's evidence-grounding requirement.

The empty intervention uses the same answer-stage instructions as the fresh
full/removal variants,including “answer UNKNOWN if unresolved”. The earlier
standalone no-retrieval arm has a different final instruction. Its score must
not be substituted for this empty counterfactual or silently pooled with it.
The32 exact repeated full outputs and exact subset prompts establish the
observed causal comparison boundary;they do not establish broader generalization,
human expertise,clinical authority or a validated estimator/global update.

No case is automatically reweighted,quarantined or promoted. Missing feedback
stays unobserved;completed invalid outputs are observed pipeline failures.
Exposure,conditional usefulness,reliability and rarity remain separate. All
source failures stay in the32-set denominator. The64 reserved public questions
remain unused. Train-only actual-need generation has now started,followed by
six matched trainings and independent gradient/state replay. Cross-type
calibration,poisoning/rollback,stronger/internal/combined controls and the
remaining T1/T2 research gates remain open.
