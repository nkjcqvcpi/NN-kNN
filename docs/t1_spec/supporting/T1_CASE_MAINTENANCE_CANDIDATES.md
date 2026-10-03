# T1 case maintenance: candidate measures after the survey

Date: 2026-09-20. Status: PI-proposed measures to compare, with two confirmed definitions. This note supersedes instructions to combine Q and B in one score. It does not select a final maintenance formula or implement a new algorithm.

The PI's original notes and clarification replies are preserved in [the source record](../resources/research/case-maintenance-2026-09-20/pi-design-notes.md). Literature definitions remain in [the survey](../05-evidence-and-citations/CBR_CASE_MAINTENANCE_SURVEY_2026-09-17.md). Here, “section 4” means the coverage-to-reachability ratio, and “section 12” means EnergyCompress.

## The proposed comparison

| Candidate | What it measures | Status |
|---|---|---|
| Q, or a variant based on C/H | Successful versus unsuccessful outcomes associated with a case's use, weighted by activation | Active candidate on its own |
| B | Information about case usefulness represented by its trained bias | Alternative to Q; do not combine Q and B in the current comparison |
| Coverage-to-reachability ratio | How many reference queries a case solves, relative to how many cases solve its own problem | Active candidate using the confirmed solve relation below |
| Influence measured by case removal | Change in final prediction loss when the case is unavailable | Active candidate; full-query and cached-query versions, with optional retraining |
| Similarity between activation maps | Whether cases receive similar activations on the same queries | Deferred at the PI's request |

The maximum case count K and existing protection requirements remain in force. Compare quality and case-base size; the acceptable loss increase and final removal/stopping policy remain to be specified.

## 1. Q and B are alternatives

Keep the existing C/H mechanism. For an evaluated query q, let s(q) be 1 for a successful final prediction and 0 otherwise. When adaptation is enabled, use the adapted prediction. Let a_i(q) be case i's normalized activation. These activations are computed from the learned model; they are not themselves learned parameters.

```text
C_i = sum_q a_i(q) * s(q)
H_i = sum_q a_i(q) * (1 - s(q))
A_i = C_i + H_i
Q_i = (C_i + s) / (C_i + H_i + 2s), with smoothing constant s > 0
```

The constant s in Q is the earlier smoothing constant, distinct from the query outcome s(q). The PI accepts Q itself or a variant as a candidate. This does not restore the earlier product with B. Activation weighting distinguishes strong participation from a small contribution; ordinary good/bad vote counts do not make that distinction.

B is a separate candidate derived from the learned case bias b_i. Its transformation and comparison groups must be specified. The PI expects Q and B to overlap because both reflect experience with the same prediction task. Their correlation is a hypothesis to measure, not an established identity. The current design therefore compares Q and B separately rather than multiplying them with exponent weights. Logging both for analysis is compatible with using only one to rank cases.

Q measures outcome quality relative to use. A high Q does not by itself imply broad coverage: a case used successfully on a few queries can have a similar Q to one used successfully on many queries. A_i and the number of queries served retain that exposure information. A large trained bias may accompany broad usefulness, but it also depends on the learned distance, competing cases, regularization, and training history. Neither Q nor B directly measures how much the remaining cases could replace a case.

## 2. Coverage-to-reachability ratio

**Confirmed definition:** a case solves a query if its normalized activation exceeds a threshold and the final query outcome is successful. When adaptation is enabled, success is judged after adaptation, exactly as in C/H. Its stored label need not agree with the query label.

Let S be the current active case base, theta the model checkpoint, D a fixed set of evaluated queries, and tau an activation threshold. For the binary success definition:

\[
\operatorname{solve}(i,q)=\mathbf 1[a_i(q)>\tau]\,s(q).
\]

The notation \(\mathbf 1[\cdot]\) equals 1 when the condition is true and 0 otherwise. Define:

\[
\operatorname{Cover}(i)=\{q\in D:\operatorname{solve}(i,q)=1\},\qquad
\operatorname{Reach}(q)=\{j\in S:\operatorname{solve}(j,q)=1\}.
\]

Let q_i denote the query corresponding to case i's problem and reference answer. The proposed ratio is

\[
M_i=\frac{|\operatorname{Cover}(i)|}{|\operatorname{Reach}(q_i)|}.
\]

The numerator counts queries served successfully by i above the threshold. The denominator counts other cases that solve i's own problem. For example, 12 covered queries and 3 solvers of i's problem give M_i = 4. This reproduces the form of the coverage-to-reachability ratio in survey section 4, using a neural solve relation.

When a query is itself a stored case, exclude that same case from retrieval before computing its activations and outcome. Thus i cannot establish that its own problem is easy to solve merely by retrieving itself. The reference collection must contain, or separately evaluate, q_i for every case being scored. Fix theta, S, the query collection, the success criterion, and tau when comparing these ratios.

A successful query can have no declared solver if all its activations fall below tau. A zero denominator is also possible. The threshold and zero-denominator policy remain open; do not silently replace zero by an epsilon or automatically protect/delete the case.

The ratio considers coverage and alternatives, but its denominator concerns i's own problem. It does not count alternative solvers separately for every query in Cover(i), as relative coverage in survey section 3 does. Nor does it directly test the loss caused by removing i. This distinction should remain visible in the comparison.

## 3. Influence measured by case removal

The PI proposed the name **case influence score**. **Case removal score** is a suggested more specific name, because it identifies the intervention being measured. The naming choice does not change the definition below.

**Confirmed loss:** use final prediction loss only. Report the adaptation penalty separately; do not add it to this score. With adaptation enabled, evaluate the final adapted prediction in both conditions. The loss can be the chosen classification or regression prediction loss, which must be declared for each experiment.

Let \(\ell_\theta(q;S)\) denote this loss on query q using model parameters theta and case base S. Its mean over the fixed reference set is

\[
L_D(\theta,S)=\frac1{|D|}\sum_{q\in D}\ell_\theta(q;S).
\]

The score of case i without retraining is

\[
I_i=L_D(\theta,S\setminus\{i\})-L_D(\theta,S).
\]

Positive I_i means removing the case increases mean loss. Negative I_i means removal improves it. Zero means no change in the mean; individual queries can still change. Remove the lowest-scoring eligible case first. If all scores are positive, the lowest identifies the smallest immediate loss increase among those candidates. Whether to accept that increase depends on the case budget and the allowed performance loss.

After each actual removal, recompute the relevant predictions and scores for the new case base. Removing several cases using scores calculated before any removal can discard cases that replace one another. A sequence of individually small loss increases must also be checked against the original pre-maintenance performance, since the increases can accumulate.

The frozen-parameter comparison follows these steps:

1. Save the model checkpoint, active case base, reference queries, baseline predictions, and losses.
2. Temporarily remove the candidate case i from retrieval.
3. Recompute retrieval, normalized activations, and reuse using the remaining cases. Run adaptation again when enabled.
4. Measure final prediction loss on the same reference queries and subtract the baseline loss.
5. Restore i before testing another candidate against that same baseline.

The phrase “without the query” in the PI's original notes is interpreted as removing the candidate case being scored. If q is also stored in the case base, ordinary leave-one-out excludes q's own stored case in both conditions. The removal condition additionally excludes i. The reference query remains in D. These are two separate exclusions, not two different reference sets.

**Keep previously learned knowledge.** The PI explicitly permits the model to retain parameters trained using a removed case. The comparison asks whether the explicit case is still needed given what the model already knows. It measures case knowledge influence at inference. It does not erase the case's earlier model parameter influence, and such erasure is not the maintenance objective.

## 4. Make removal scoring cheaper with cached queries

During forward passes, record the queries on which each case's activation exceeds a threshold, together with the activations and baseline losses. For candidate i, let

\[
D_i^\tau=\{q\in D:a_i(q)>\tau\}.
\]

The proposed inexpensive comparison reruns only these queries with i removed. A corresponding approximation to the full score is

\[
\widetilde I_i=\frac1{|D|}\sum_{q\in D_i^\tau}
\left[\ell_\theta(q;S\setminus\{i\})-\ell_\theta(q;S)\right].
\]

Use the full denominator |D| when approximating the full mean-loss change. Dividing by |D_i^tau| would instead measure the average effect conditional on i being sufficiently active, and would change the comparison between frequently and rarely used cases. The activation threshold selects queries to rerun; do not multiply the loss difference by activation again unless deliberately defining another score.

This avoids a full reference-set evaluation for every candidate. Baseline losses can be reused within the same checkpoint and case-base state. The index maps each case ID to its query IDs; retain enough query data to rerun prediction, not just the activation values.

**When is it exact?** It is exact if removing i cannot change prediction on any omitted query. Exact zero activation can support this condition when the remaining retrieval and reuse computation is unchanged. An ordinary positive threshold does not establish it. Dense softmax usually assigns small nonzero weights to omitted cases, and removing one changes the normalization of the others. A small activation alone also does not guarantee a small loss change after regression or neural adaptation. Treat thresholded scoring as an approximation and compare it with full-query scoring on a manageable evaluation sample.

The cache must identify the model checkpoint and active case-base state. Training updates or case removals can change which queries a case activates for. Historical activation records can identify likely affected queries, but they are not an exact current index unless refreshed or otherwise verified. Refresh the index and baseline losses at maintenance checkpoints and after case-base changes as required by the chosen approximation. A fresh forward pass shared across candidates can still be much cheaper than one full pass per candidate removal.

## 5. Optional retraining after removal

The PI proposes a higher-cost version that lets the remaining model adjust after case removal. This may permit further compression, but the improvement and additional computation must be measured.

Starting from the saved checkpoint, remove i, then train under a declared budget and reevaluate. Let theta_i' be the resulting parameters. A practical score is

\[
I_i^{\mathrm{retrain}}=L_D(\theta_i',S\setminus\{i\})-L_D(\theta,S).
\]

This compares removal plus further training against the starting model. To distinguish the benefit of removal from the benefit of extra training, also continue training the unmodified case base for a matched budget. Start each candidate experiment from the same saved state; do not carry one candidate's training into another candidate's test.

Retraining can change predictions on queries where i previously had zero activation. The cached-query restriction is therefore not generally valid for this branch. Evaluate its final effect on the full designated reference set, or use a separately justified sampling estimate. The training budget, parameter scope, and whether to start from the existing checkpoint or train afresh remain experimental choices. Continuing from the existing checkpoint is consistent with the PI's permission to keep knowledge learned from removed cases.

## 6. Activation maps: preserve the idea, defer its use

For a common ordered collection of queries \(q_1,\ldots,q_m\), case i's **activation map** is the vector \((a_i(q_1),\ldots,a_i(q_m))\). This records where the case activates and by how much. Similar activation maps could identify cases that participate on similar queries at similar strengths.

The PI identified a failure mode: two duplicate cases can receive very different maps if training favors one and suppresses the other. The reverse concern also matters: two cases with similar maps can supply different solutions or complementary information. Therefore map similarity alone does not establish redundancy. Preserve the query/activation records for the other measures, but defer a redundancy score based on map similarity. Do not introduce a map-distance threshold or automatic duplicate deletion now.

## 7. Relationship to backpropagation and prior maintenance methods

NN-kNN already learns representations, feature weights, and case biases through backpropagation. Its neural adapter also has trainable parameters. Which components are updated together depends on the training phase. This provides a richer model-learning mechanism than holding a similarity measure fixed while selecting cases, as EnergyCompress does in its reported setting.

Learning these parameters does not itself calculate the effect of case removal or select the discrete active case base. It is therefore not yet a reproduction of EnergyCompress. The proposed removal experiment supplies that missing measurement. Whether broader neural learning produces better maintenance is an empirical question; differentiability alone does not establish superiority.

The PI's intended contribution is to recover earlier methods as special cases or reproduce their operations within full-cycle neural CBR. The following distinguishes established connections from remaining work:

| Prior method or family | Connection to the proposed measures | What still has to match |
|---|---|---|
| Relative coverage | Uniform credit among successful solvers recovers the earlier C = RC, H = 0 connection. | The solver relation, reference queries, and zero activation outside the solver set; full RC-CNN reproduction also needs its selector. |
| Coverage-to-reachability ratio | The same count ratio with NN-kNN's thresholded solve relation. | Matching the historical solve relation gives the same score; using the confirmed neural relation gives a neural variant. |
| Reputation / correct-incorrect evidence | C/H offers activation-weighted outcome evidence. | Historical reputation may use each case's label agreement and unit event counts, rather than a shared adapted query outcome. |
| EnergyCompress | Loss after removal minus loss before removal. | Its specific energy, hinge loss, margin, and final selection procedure. Another final prediction loss gives a related removal method. |
| RelCBR | Joint learning of case-related weights offers a connection to B and the trainable model. | RelCBR's reliability-weighted predictions, weighting of target errors, objective, and constraints; backpropagation by itself is not score equality. |
| Adaptation effort and cooperative competence | Full-cycle neural CBR can evaluate reused solutions and combinations of cases. | Measures of adaptation cost, explicit adaptation-rule knowledge, or required case combinations are additional mechanisms, not already represented by Q or B alone. |
| Drift, protection, and partial-case maintenance | The broader system can retain histories, protect cases, and support additional maintenance operations. | Temporal windows, confidence rules, activation/deactivation policies, coverage checks, and deletion of parts of cases must be reproduced explicitly. |

These connections organize the research program. They do not yet establish that every surveyed algorithm is a special case of one completed formula. Preserve the distinction between reproducing a score and reproducing the full maintenance procedure.

## Remaining design choices

- Activation thresholds, reference-query selection, task success criteria, and the ratio's zero-denominator policy.
- Q variant and smoothing; the B transformation and comparison groups.
- Final prediction loss for each task. Its scope is settled: exclude the adaptation penalty and report that penalty separately.
- Acceptable cumulative loss increase, case budget, score-update schedule, and comparison of memory saved with computation spent on maintenance.
- Cache refresh and validation of approximation error; optional retraining budgets and matched further-training controls.

The final test set remains separate from the queries used to select cases. The PI's phrase “rerun test” is interpreted as evaluating the designated maintenance reference queries, not repeatedly selecting cases against the final test set.
