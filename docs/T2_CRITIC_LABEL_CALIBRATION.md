# T2 optional critic labels and activation matching

The pilot CLI now exposes existing fixed/mutable/trainable/hybrid label modes,
raw activation matching threshold and EMA alpha. Nonfixed modes require an
explicit NN-kNN critic before output creation. Fixed remains the default; no
learner change or fabricated value prior is introduced.

24 original Acrobot runs plus24 full actual retrainings use8,192 interactions,
seeds8/9/10,NN/NN,hard target interval1,capacity128/top-k8,preserved-by-ID Adam,
no scheduled bias pruning,no early stopping. Threshold0 compares all four modes;
thresholds−.25/−1 compare mutable/hybrid,alpha.25 throughout. These are exploratory
diagnoses,not confirmation or Stage A success.

Actual replay reconstructs225 batches/196,608 GAE samples/843 raw-positive actor
recommendations;150 mutable calls update5,669 rows from36,248 matched samples.
The remaining160,360 samples append critic cases. Core retrieval distances,
raw activation masks,weighted aggregate targets and one EMA update per matched
row agree exactly.1,634 Adam moment tensors/4,474 other optimizer states retain
stable-ID alignment. All final actor/critic/target/optimizer/evaluation and
independent MC states reproduce; three fixed controls match the previous startup
pilot exactly.201 training snapshots/468 events/all3,744 fixed-set removals and
192 independent MC events/all1,536 removals independently replay.

| Condition | Seeds8/9/10 greedy returns | First actual ready behavior |
|---|---|---|
| Fixed or mutable,threshold0 | −500/−500/−458 | 7001/7001/unavailable |
| Trainable or hybrid,threshold0 | −500/−110.667/−458 | 7001/7001/unavailable |
| Mutable,threshold−.25 | −500/−500/−500 | 8001/7001/unavailable |
| Hybrid,threshold−.25 | −500/−370/−500 | 8001/7001/unavailable |
| Mutable or hybrid,threshold−1 | −458/−458/−500 | unavailable/unavailable/8001 |

At threshold0,no mutable update executes. Three fixed/mutable and three
trainable/hybrid checkpoint-state pairs match exactly,including optimizer
states. The trainable-label improvement is limited to seed9 and does not justify
a general recommendation. At−.25,seed10 satisfies final readiness only after
final insertion: no ready-policy behavior/gradient evidence exists during its
training. At−1,seeds8/9 never become ready;−458 is uniform fallback,not learned
improvement over−500. Critic prediction and actor coverage have distinct effects.

24 common-policy diagnostics use the baseline actor and baseline time-limit
bootstrap; all eight bounded query/target pairs,target means,sample counts,
returns and bootstrap counts match. Threshold−1 substantially lowers these MSEs
(seed8 roughly4048→2084,seed9 1015→496,seed10 3982→1928) despite actor coverage
failures. This evaluates learned V under a different policy: it is an off-policy
transfer diagnostic,not proof of on-policy calibration. Own-policy MC populations
can differ. No diagnostic data enter gradients,label matching or case admission.

Frozen label-mode source
`cb51041421800574e0193514fdc78027528cdb256af10aa11c07f30cb48de347`,
ZIP SHA256 `d727a1cf024d1e9c1c2e381ce97a53c65e1bb1831f2338a32d3a5291f67cb922`;
matching/replay source
`88fafd106c63d461d4e196e9f8e0272a83c2dd3cf7196ac674ef4371c124584f`,
ZIP SHA256 `66e3eaecf2342729bd9d5dd002f4536efb77ec05353b0ed36f78438459f53aab`.
Both132-file archives and ZIP bytes validate; only `tools/t2_role_pilot.py`
differs. Existing187-test learner boundary and six current focused optimizer,
identity and training-audit checks pass; CLI preflight and compilation pass.
An initial report accidentally included generated cache files in cross-source
comparison; excluding caches follows the frozen-source manifest and verifies
all actual source files. A common-policy helper initially used the older2,048
global step; its failed check is preserved,then corrected to8,192 and rerun.

Task evidence: `work/critic_label_modes_8192/`, `work/critic_matching_{m025,m1}/`,
`work/label_matching_replayed/`, and
`outputs/neural-cbr-t2-label-matching-{results.md,verification.json,training-verification.json,mc-verification.json,report-verification.json}`;
common-policy evidence `outputs/neural-cbr-t2-label-fixed-policy-verification.json`.
Broader matched cost/500-case budgets,actor utility and full-cycle Stage A gate
remain open. Imported18 authority documents are unchanged.
