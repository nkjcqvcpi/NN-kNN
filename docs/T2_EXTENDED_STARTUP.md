# T2 Acrobot startup at 2,048 and 8,192 interactions

Eighteen original runs and eighteen full actual retrainings: NN/NN,three seeds,
two strict interaction budgets,EMA/hard/none targets,interval1,capacity128/top-k8,
ID-preserved Adam,no scheduled bias pruning,no early stopping. Default fixed GAE
critic labels and staged raw-positive actor recommendations are unchanged.

Every original GAE computation and critic/actor admission is reconstructed from
actual observations/actions/rewards/boundaries/target values.111 batches/92,160
training samples/351 raw-positive samples verify; actor admission matches the
RAW advantage mask exactly. All final actor/critic/target/Adam/evaluation and
independent MC states reproduce. Nine short/long first2,048 observation/action
prefix hashes match.93 training snapshots/216 events (186 GAE,30 actor signed
surrogate)/all1,728 fixed-set removals replay;18 final-state MC audits144
queries/all1,152 removals replay. SourceZIP132 files verifies.

| Budget | Seed | First actual ready step | Final action counts | Greedy return |
|---|---:|---:|---|---:|
| 2,048 | 8 | unavailable | [0,0,0] | -458 fallback |
| 2,048 | 9 | unavailable | [2,1,0] | -458 fallback |
| 2,048 | 10 | unavailable | [0,0,0] | -458 fallback |
| 8,192 | 8 | 7,001 | [2,3,4] | -500 learned policy |
| 8,192 | 9 | 7,001 | [42,24,30] | -500 learned policy |
| 8,192 | 10 | unavailable | [1,4,4] | -458 fallback |

Counts/readiness/returns agree across the three target modes in this pilot,
while critic parameters differ. Seeds8/9 have only1,192 behavior interactions
after readiness; seed10 never satisfies action coverage. Proportional exploration
durations differ by budget, but no short run becomes ready and every long first
ready step occurs after its4,096-step schedule,at epsilon.05. Short final partial
rollout updates differ from the long run's continuing episode; equal experience
prefix is not environment/RNG/checkpoint resumption.

Readiness is not policy effectiveness. Seed8's nine raw-positive recommendations
have no true-terminal positive positions. Positive GAE may reflect value
differences/calibration error on negative-reward trajectories, not certified
success. Seed9 has96 positives,including five true-terminal positions,but its
greedy policy still fails. Never replace raw advantage with normalized advantage
or admit nonpositive recommendations just to manufacture coverage.

All six hard/none final critic parameter states match exactly at interval1:
the hard copy is current online value for the next batch,with no intervening
gradient update. For all12 EMA/hard runs active fixed-label values equal online
and target (distinct storage); existing-ID fixed labels cannot exhibit label
learning lag. Compare active IDs only: inactive target rows can retain lagged
values without contributing. The first report check incorrectly compared inactive
rows; corrected active-only check and the failed attempt are preserved.

Values remain optimistic relative to negative independent MC targets. The next
explicit experiment compares fixed/mutable/trainable/hybrid labels,reporting
whether mutable raw-activation matching actually executes; this does not inject
fabricated priors. Broader budgets,500-case defaults,matched cost/coverage and
Stage A thresholds remain open. This is diagnosis,not full-cycle success.

Frozen source `dc26517dfadb6d13be47acd74e974d1a3444ce75b848431e97a3ecadbfda9add`;
132 files; ZIP SHA256
`c0828c5fabe1bf5b9571372a915a46d93d3f0d72d3fd7860d4094bb898d02ba0`.
Existing187-test implementation boundary remains; no learner change in this
package. Evidence:task `work/acrobot_startup_{2048,8192}/`, `work/startup_replayed/`,
`outputs/neural-cbr-t2-startup-{results.md,verification.json}` and separate
training/MC verification JSONs. Imported18 authority documents remain unchanged.
