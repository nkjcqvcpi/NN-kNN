# CBR case maintenance: criteria, scores, and selection procedures

Survey date: 2026-09-17. Status: literature survey for discussion, not an adopted NN-kNN formula or an implementation specification.

Expanded 2026-09-19 at the PI's request: definitions, notation, score interpretation, worked examples, and connections between methods. The examples and mathematical connections explicitly marked as explanatory are our derivations, not reported experiments or newly adopted maintenance formulas.

**Numbering:** Method-section numbers and source IDs are different. Section 11 explains RelCBR (source S12); section 12 explains EnergyCompress (source S15). The headings below identify both where this distinction caused confusion.

The PI supplied the IJCAI-2018 survey and IJCAI-2025 EnergyCompress paper. This focused review follows their references and adds relevant work on reliability, adaptation, and changing workloads. It is not a systematic review with an exhaustive search protocol. Sources are linked below. Public PDFs retrieved for this review are preserved in `resources/research/case-maintenance-2026-09-17/`, with provenance and SHA-256 checksums in its manifest and `resources/README.md`.

## Scope and decisions

- **Post-survey candidates, 2026-09-20:** The PI proposes Q or a C/H variant, B alone, coverage-to-reachability, and case-removal influence. Q and B are alternatives, not factors in a product. The [candidate design note](../03-research-architecture/T1_CASE_MAINTENANCE_CANDIDATES.md) defines these measures, the cached-query approximation, optional retraining, and the deferred activation-map idea. It records both confirmed choices: solve uses the final adapted query outcome, and removal influence uses final prediction loss only, with the adaptation penalty reported separately.

- **Generalization objective approved 2026-09-20:** Devise a general case maintenance score and method, then establish which earlier maintenance approaches it recovers as special cases. The relative-coverage reduction is the first component-level connection. The complete formulation and further reductions remain research work.

- The 2026-09-17 survey set aside the previous provenance-quality and geometric-combination formulas. On 2026-09-20, the PI reintroduced Q or a variant as a standalone candidate. The Q/B combination remains superseded; no final comprehensive score is selected.
- Call the eventual broad measure the **case maintenance score**. This is the project's chosen name, not a claim that the literature uses one standardized score under that name.
- Treat coverage and redundancy as one maintenance consideration. Evaluate a case's coverage relative to the other cases retained.
- Survey first, then brainstorm. The literature survey does not itself approve a replacement formula or selector. Subsequent PI decisions on C/H are recorded below. The existing maximum case count `K` and protected-case requirements remain in force.
- **C/H rule approved 2026-09-20:** When cases contribute to a combined adapted prediction, assign that query's outcome to all participating cases in proportion to their normalized activations. Accumulate the evidence over queries. This makes maintenance consider retrieval and reuse together. The comprehensive case maintenance score remains to be devised.

## What maintenance must specify

CBR follows retrieve-reuse-revise-retain. Maintenance governs which case knowledge remains available and how the store is updated. A complete maintenance method needs more than a scalar rank. Wilson and Leake's framework distinguishes data collection, triggering, available operations, and execution, and considers coordination among representation, similarity, adaptation, and case knowledge [S3].

For NN-kNN, the following is our synthesis of the reviewed literature and existing project requirements:

| Specification | What must be decided | Relevant literature |
|---|---|---|
| Task competence | What counts as solving a problem: correct class, acceptable regression error, or successful adapted solution? | S2, S4, S8, S9 |
| Coverage with redundancy | Which problems lose adequate support after removal, given the remaining cases? | S4, S5, S10 |
| Case fidelity and harmful use | Is the stored solution credible, and does using it damage predictions? These are distinct questions. | S5, S11, S12, S13 |
| Reuse and adaptation | Does a case enable inexpensive correction or provide knowledge needed to adapt other cases? | S7, S8 |
| Interactions | Does competence require several cases jointly? Can other cases replace one being removed? | S9, S10 |
| Workload and time | Which reference problems represent future use? How are changing concepts distinguished from noise? | S7, S10 |
| Resource cost | Is the limit case count, bytes, retrieval time, adaptation time, or maintenance computation? | S2, S7, S14 |
| Evidence and action | What evidence suffices, when are scores recomputed, and when is removal reversible or reviewed? | S3, S10, S12; existing project rules |

Some items belong in a score; others can be constraints or procedural rules. The reviewed papers do not establish a single formula covering all of these for a jointly learned neural retriever and adapter.

## Reading guide

Start with the notation and terminology below. Sections 3 and 4 compare two ways to combine coverage with redundancy. Sections 5, 10, 11, and 13 address harmful or unreliable cases. Sections 6-8 add adaptation and cooperation. Section 12 explains removal-loss scoring. The [comparison tables and worked connections](#comparing-and-connecting-the-methods) bring these strands together.

## Shared notation and terminology

The notation is standardized to make the papers comparable. It does not make their underlying definitions identical. In particular, "solves," "neighbor," "competence," and "support" must be interpreted within each method.

### Objects and mathematical symbols

| Symbol | Meaning in this note |
|---|---|
| \(C\) | The original or candidate case pool. |
| \(S\subseteq C\) | The currently retained case set. Membership changes during maintenance. |
| \(c=(x_c,y_c)\) | A stored case: problem description \(x_c\) and stored solution \(y_c\). The solution may be a class, a number, or a structured object. |
| \(q=(x_q,y_q)\in D\) | A reference problem and the answer used to assess performance. "Solve \(q\)" means predict from \(x_q\), then compare with \(y_q\). Writing \(f_S(q)\) is shorthand for \(f_S(x_q)\). |
| \(D\) | The designated reference set for calculating scores. Some papers use the original case pool; a separate reference set is also possible. Its membership must be specified independently of the shrinking \(S\). |
| \(f_S(x)\), \(\widehat y_q\) | The reasoner's prediction using \(S\). Retrieval, voting, and any adaptation are part of the declared procedure. A hat indicates an estimate. |
| \(N_k(q;S)\) | The \(k\) nearest eligible cases to \(q\) in \(S\), under the stated distance or similarity. Abbreviated to \(N_k(q)\) when \(S\) is fixed. |
| \(k\) versus \(K\) | Lowercase \(k\) is a neighbor count. Uppercase \(K\) is our active-memory capacity. Neither is the smoothing function called \(K(\cdot)\) in S8. |
| \(d(c,j)\), \(s^P(c,j)\), \(s^Y(c,j)\) | Problem-space distance, problem-space similarity, and solution-space similarity. Smaller distance means closer; larger similarity means more similar. They are not automatically normalized. |
| \(\lvert A\rvert\), \(\varnothing\), \(\{c\}\) | Number of elements in a set, the empty set, and the set containing only \(c\). |
| \(A\setminus\{c\}\), \(A\cap B\), \(A\cup B\) | Remove \(c\) from \(A\); intersection; union. A union counts a shared element once. |
| \(\sum_{j\in A}\), \(\mathbf1[P]\) | Sum over elements of \(A\); an indicator equal to 1 when condition \(P\) is true and 0 otherwise. An empty sum is 0. |
| \(\forall\), \(\exists\), \(\arg\min\) | "For every," "there exists," and the argument attaining a minimum. A minimum gives a value; an argmin identifies the case or outcome attaining it. Ties require a rule. |
| \(e\), \(\ell\), \(L_D\) | A prediction error, a per-query loss, and an aggregate loss over \(D\). Their exact definitions differ between methods. Lower is better. |
| \(t\) | An update or scoring iteration where used as a subscript. It is not automatically a case's age. |

**Self-retrieval and leave-one-out.** When a stored case is used as a reference query, leave-one-out excludes that same case from the memory used to predict its answer. Exact self-matching can otherwise inflate measured competence. This differs from a removal experiment, which excludes the candidate being scored. When both procedures are used, both exclusions apply. The papers' exact protocols still govern reproduction.

**Reference answer versus verified truth.** The symbol \(y_q\) denotes the answer available to the scoring procedure. It does not guarantee that the answer is correct. This matters especially for reliability methods inspecting potentially corrupted case bases.

### Coverage, reachability, and necessity

First define a method-specific binary relation \(A(c,q)\): it equals 1 when the method regards case \(c\) as able to solve, or as successfully contributing to solving, reference problem \(q\). These interpretations differ; the individual sections state which applies.

For a fixed relation, the two directions are:

\[
\operatorname{Cov}_{S,D}(c)=\{q\in D:A(c,q)=1\},\qquad
\operatorname{Reach}_{S}(q)=\{c\in S:A(c,q)=1\}.
\]

Coverage asks, **"Which problems does this case solve?"** Reachability asks, **"Which cases solve this problem?"** They are the outgoing and incoming sides of the same relation. An arrow \(c\to q\) represents a successful relation, not merely geometric closeness.

- In single-case adaptation models, \(A(c,q)\) can mean that adapting \(c\)'s solution solves \(q\) within the allowed effort.
- In S5's classification model, it requires a correct aggregate prediction, retrieval of \(c\), and label agreement.
- In compositional adaptation, several cases may be needed together. A pairwise relation alone cannot record that requirement; Section 8 adds supporting sets.

The **coverage set** is the collection of problems. Its **coverage count** is the number of those problems. Neither is automatically an accuracy, a probability, or a measure of unique contribution.

**Redundancy is relative to the task and retained set.** Cases are functionally redundant when the remaining cases can supply their capability. Similar input descriptions do not prove redundancy. **Necessity** asks whether removal loses capability in the present set. Relative coverage divides shared credit; it does not directly test necessity.

### Terms that recur across papers

| Term | Definition for reading this survey |
|---|---|
| Competence | Ability to solve specified problems. Older methods often use binary solvability; EnergyCompress uses negative loss. Check the operational definition before comparing scores. |
| Fidelity | Correctness or quality of the stored problem-solution association. It concerns the case's content. |
| Reliability | Estimated confidence in a case under a stated model or evidence procedure. It is related to fidelity, not verified truth. |
| Utility | Benefit under a specified objective, such as expected time saved or prediction improvement. The word alone does not identify an equation. |
| Adaptation | Changing or combining retrieved solutions during reuse. Adaptation cost measures effort; adaptation error measures discrepancy from the reference answer. |
| Knowledge containers | CBR knowledge in the case base, vocabulary/representation, similarity model, and adaptation knowledge. A case can contribute to more than one container. |
| Competence preservation / enhancement | Keeping existing problem-solving capability while reducing memory / improving capability, for example by removing harmful cases. These are objectives, not guarantees on unseen data. |
| Condensing / editing | Selecting representative cases for a smaller memory / modifying the instance set, often to remove noisy or redundant cases. Usage varies by paper. |
| CNN / RC-CNN / RP-CNN | Condensed Nearest Neighbor, and variants that order candidates using relative coverage or relative performance. Here CNN does **not** mean convolutional neural network. |
| Footprint | A compact subset intended to preserve competence under the chosen solving model. It need not be a globally smallest subset. |
| Class boundary | A region where nearby inputs can have different labels. A boundary case may be informative despite considerable neighborhood disagreement. |
| Concept drift | A change over time in the distribution of problems or their relationship to solutions. A previously useful case can become inappropriate without having been incorrectly recorded. |
| Score / selector / trigger | A numerical assessment / the rule choosing cases to add or remove / the condition that starts maintenance. Different selectors can use the same score. |

## Landscape and formulas

All numerical examples below are constructed for explanation, not benchmark results.

The shared notation applies below; each method's specific solving relation and neighborhood still govern its interpretation.

### 1. Survey map: Juarez et al. (2018)

The supplied survey separates nearest-neighbor editing from competence-based maintenance and reviews extensions involving case structure, time, and solution value. It is useful as a map of methods, not as a universal scoring specification [S1, Sections 2-3].

Verification caution: its Section 2.3 prints `RC(c,C)=1/|RS(c,C)|` and an increasing-order procedure. The original compact-case-base paper instead defines a sum over covered problems and describes decreasing relative-coverage order [S4, Definition 8 and Section 3.3]. Use the original definition below when implementing RC-CNN. The two displayed expressions are not interchangeable.

### 2. Competence-preserving categories: Smyth and Keane (1995)

This work distinguishes cases by their role in preserving competence, including pivotal, auxiliary, spanning, and support cases. A pivotal case contributes capability that would be lost on deletion. The maintenance policy uses these roles instead of relying solely on usage or speed savings [S2, Sections 2-3].

The competence categories have different operational meanings:

| Category | Meaning in the paper's adaptation-based model |
|---|---|
| Pivotal | No other case solves the case's own problem: \(\operatorname{Reach}(c)\setminus\{c\}=\varnothing\). Deletion leaves at least that problem unsolved. |
| Auxiliary | Another reachable case provides strictly broader coverage, subsuming this case's coverage. Keeping the auxiliary case may still save effort. |
| Spanning | Its coverage connects regions also covered by other cases. Its importance can increase when cases in those regions are deleted. |
| Support group | Members provide the same coverage and can stand in for one another. Removing the whole group can lose capability if no outside cases replace it. |

These are structural roles, not four numerical score ranges. A "support group" supplies alternatives with shared coverage. Section 8's "support cases" are instead required together. The paper also discusses an earlier utility expression:

\[
U(c)=\operatorname{ApplicationFreq}(c)\operatorname{AverageSavings}(c)-\operatorname{MatchCost}(c).
\]

**Reading the terms.** Application frequency describes how often the case is used. Average savings is the mean effort saved per use relative to the chosen alternative problem-solving process. Match cost is the overhead of considering or matching the case. These need a common accounting basis: expected uses times cost saved per use, minus the corresponding workload overhead. Positive \(U\) means net savings, not unique competence.

**Example.** With 10 expected uses, 3 cost units saved per use, and 8 units of overhead on that workload, \(U=10(3)-8=22\). A uniquely necessary case used only once could score lower.

This is a performance-benefit measure, not the paper's complete competence-preserving policy. Its warning is pertinent: an infrequently used case can be indispensable when the system has no other way to solve its problems. Category definitions depend on the solving relation and the remaining memory.

### 3. Relative coverage: Smyth and McKenna (1999)

Each solved problem contributes credit divided among its possible solvers:

\[
RC(c)=\sum_{q\in\operatorname{Cov}(c)}\frac{1}{|\operatorname{Reach}(q)|}.
\]

**Reading the formula.** For each \(q\) that \(c\) covers, count all cases that solve that same \(q\). Give \(c\) the reciprocal of this count. Sum the credits over its coverage. The denominator concerns \(q\), not \(c\)'s own problem. Every denominator is positive under a consistent relation because \(c\) itself is one of \(q\)'s solvers.

**Example.** If \(c\) covers three problems having 1, 2, and 4 solvers, \(RC(c)=1+1/2+1/4=1.75\). A different case covering three problems with 10 solvers each receives \(0.3\). Equal coverage counts need not mean equal scores.

**Scale.** RC is nonnegative and at most \(\lvert\operatorname{Cov}(c)\rvert\). It can exceed 1. It is a sum of shared problem credits, not a probability.

Broad coverage raises the score; alternative solvers reduce each problem's credit. This directly combines coverage with redundancy. RC-CNN orders cases by decreasing score, then constructs a condensed set using solvability checks [S4, Definition 8, Section 3.3].

**Our interpretation:** RC allocates shared credit. It is not identical to the loss caused by removing a case. If two cases duplicate all their coverage, both can have positive RC even though either one can be removed without losing that coverage. A one-time ranking therefore cannot replace the selection procedure.

The full PDF download failed in this session. Definition 8 and ordering were checked through the indexed original-paper text and cross-checked against S13, Equation 10.

#### Connection to NN-kNN activation and the earlier C/H components

The PI raised this connection on 2026-09-19. The following is an algebraic comparison with the [documented NN-kNN activation and classification components](../03-research-architecture/T1_REVISE_RETAIN.md), not a demonstrated superiority claim or a selected maintenance formula.

**Both distribute credit per query.** In the normalized NN-kNN formulation, case activations are nonnegative and sum to 1 over the eligible cases for each query. For example, softmax turns raw retrieval scores into weights:

\[
a_i(q)=
\frac{\exp(s_i(q)/\tau)}
{\sum_{j\in S_q}\exp(s_j(q)/\tau)},\qquad
a_i(q)\ge0,\quad\sum_{i\in S_q}a_i(q)=1.
\]

Here \(S_q\) is the set eligible for this query after any masking, \(s_i(q)\) is a raw retrieval score such as bias minus learned distance, and \(\tau>0\) is the softmax temperature. Cases outside \(S_q\) receive zero weight. Representations, feature weights, and case biases are learned; activations are computed for the query. The normalization statement concerns these weights, not the raw scores.

Relative coverage already has a per-query allocation:

\[
w_i^{RC}(q)=
\begin{cases}
1/\lvert\operatorname{Reach}(q)\rvert,&i\in\operatorname{Reach}(q),\\
0,&\text{otherwise}.
\end{cases}
\qquad RC(i)=\sum_{q\in D}w_i^{RC}(q).
\]

For a covered query, the weights sum to 1. For an uncovered query, all weights are 0. RC assigns equal shares among cases satisfying its solving relation. NN-kNN instead assigns potentially unequal shares according to its learned retrieval model, including cases that may support a wrong answer.

**What the earlier classification C/H split adds.** In the retrieval-only stored-label baseline, a query contributes \(a_i(q)\mathbf1[y_i=y_q]\) to \(C_i\) and \(a_i(q)\mathbf1[y_i\ne y_q]\) to \(H_i\). For a query that cases \(a,b\) can solve and case \(c\) cannot:

| Case | Stored label agrees with reference | RC credit for this query | Example NN-kNN activation | Added to \(C_i\) | Added to \(H_i\) |
|---|---|---|---|---|---|
| \(a\) | Yes | \(0.5\) | \(0.80\) | \(0.80\) | 0 |
| \(b\) | Yes | \(0.5\) | \(0.15\) | \(0.15\) | 0 |
| \(c\) | No | 0 | \(0.05\) | 0 | \(0.05\) |

The example shows unequal credit for two successful solvers and an explicit record of opposing activation. With one stored label per case and a reference label for every query, normalization implies:

\[
\sum_i(C_i+H_i)=\lvert D\rvert,\qquad
\sum_i C_i=\sum_{q\in D}p^0_{y_q}(q),\qquad
\sum_i H_i=\sum_{q\in D}(1-p^0_{y_q}(q)),
\]

where \(p^0_{y_q}(q)=\sum_i a_i(q)\mathbf1[y_i=y_q]\) is the retrieved prediction's probability mass on the reference class. Each query contributes one unit in total, partitioned into agreeing and opposing mass.

This makes \(C_i/H_i\) more expressive about the current NN-kNN prediction than equal solver credit. It is not exactly the same quantity as RC: RC divides all credit among successful solvers, while the C/H split allocates mass across both agreeing and opposing cases. Equal activation among exactly the successful solvers, zero activation elsewhere, and agreement of the two success definitions give the special case \(C_i=RC(i)\), \(H_i=0\).

**Limits of the connection.** Per-query normalization creates competition for activation; it does not directly measure replaceability. Duplicating a case can change its group's total softmax mass, and a highly activated case can still be replaceable by another case after removal.

**Normalization with adaptation, checked 2026-09-20.** The maintained NN-kNN code normalizes case activations before prediction. Its regression adapters reuse those weights: per-case adaptation predicts \(\sum_i a_i(q)\widetilde y_i(q)\), whereas aggregate adaptation predicts \(\sum_i a_i(q)y_i+\delta(q)\). Here \(\widetilde y_i(q)\) is an adapted case solution and \(\delta(q)\) is the aggregate correction. Neither adapter changes the case weights. Thus adaptation does not remove their per-query normalization. Source: sibling `NN-kNN` checkout, commit `c09719576b3519e9878764190773916cd9ce82e6`, `model/nnknn_model.py:578-587,1036-1051,1103-1110,1167-1170` and `model/nn_cdh.py:275-285`. The current classification path also normalizes class mass before prediction; classification adaptation remains a proposed extension at this commit.

**Paper cross-check, 2026-09-20.** The archived [2026 regression paper](../resources/research/nnknn-regression/ijcai-2026-nnknn-regression.pdf), Section 3.2, Equation (6), explicitly defines nonnegative normalized weights \(\pi_{qi}\) with \(\sum_i\pi_{qi}=1\). Equations (8)-(10) use these same weights to form the aggregate case and label, then add an adaptation correction. In that paper, \(a_{qi}=b_i-d_{qi}\) is the **raw score**; its \(\pi_{qi}\) corresponds to the **normalized activation** \(a_i(q)\) in these maintenance notes. Equation (13) separately renormalizes the top-K subset for the locality loss. The earlier [2025 classification paper](../resources/research/nnknn-classification/ijcai-2025-nnknn-classification.pdf), Section 3 and Figure 1, describes independent sigmoid case gating and weighted class aggregation. The unit-mass claim here refers to the normalized model in the regression paper and current maintained code. The regression paper's Equation (15) specifies prediction, locality, and case-bias losses; the T1 penalty on unnecessarily large adaptation corrections is a proposed extension, not a term in that published objective.

**Using C/H after adaptation.** For each query, the case activations sum to one. If we count each case's activation in \(C_i\) when its contribution is helpful and in \(H_i\) when its contribution is harmful, the total added to C and H is also one. For example, a case with activation 0.2 adds 0.2 to one of these two quantities.

Let \(\Delta C_i(q)\) and \(\Delta H_i(q)\) denote the amounts added for case \(i\) on query \(q\). Then

\[
\Delta C_i(q)+\Delta H_i(q)=a_i(q),\qquad
\sum_i[\Delta C_i(q)+\Delta H_i(q)]=\sum_i a_i(q)=1.
\]

We may judge the contribution after adaptation. The activations still sum to one, so the same equation holds. Across a set \(D\) of evaluated queries, \(\sum_i(C_i+H_i)=|D|\), where \(|D|\) is the number of queries. This assumes each query has eligible cases and every case's activation is counted once across C and H. If we leave some activations out, the recorded total will be smaller.

The earlier formula for \(p^0_{y_q}(q)\) counts activations of cases whose stored labels match the correct label before adaptation. Judging contributions after adaptation can change which activations count toward C or H. Their combined total remains the same. The total added to C for a query need no longer equal that query's pre-adaptation probability for the correct class.

**PI decision, 2026-09-20: use the adapted query outcome.** When several cases contribute to a combined prediction, use the final adapted outcome to update all of them. Weight each update by the case's normalized activation. With one retrieved case, that case receives the whole update. This rule allows retention to evaluate how cases perform through retrieval and reuse together.

For a binary success/failure judgment, let \(s(q)=1\) if the final adapted prediction satisfies the task's success criterion and \(s(q)=0\) otherwise. For classification, success can mean predicting the correct class. For regression, an acceptable error threshold would define success; that threshold remains open. The updates are

\[
\Delta C_i(q)=a_i(q)s(q),\qquad
\Delta H_i(q)=a_i(q)[1-s(q)].
\]

For example, if two cases have activations 0.7 and 0.3, a successful adapted prediction adds 0.7 and 0.3 to their respective C values. An unsuccessful prediction adds those same amounts to H. All cases receive the query's outcome, with different amounts according to their activations. The total update remains one. The earlier stored-label equations remain a comparison baseline; these equations specify the accepted way to share a combined outcome. A continuous regression outcome measure and the complete case maintenance score remain open.

The PI expects evidence from many queries to make these estimates more reliable. T1 will test whether this improves maintenance as more queries are evaluated and different case groups are retrieved. If a harmful case is repeatedly retrieved with helpful cases, it can continue to receive favorable updates. The rule is therefore an accepted practical way to accumulate evidence, with its reliability to be evaluated. The proposed T1 adaptation loss also allows small task-specific corrections and penalizes unnecessarily large changes to retrieved solutions.

Some maintenance methods instead remove a case, run prediction again, and measure the change in prediction loss. Those changes measure the effect of removal. They are not case activations and do not have to sum to one.

### 4. Coverage-to-reachability ratio: Chebel-Morello et al. (2015)

**NN-kNN candidate proposed 2026-09-20:** Use the ratio below with solve defined by activation above a threshold and success of the final adapted query outcome. This is the PI's proposed neural variant, not the historical paper's success definition. See [the candidate note](../03-research-architecture/T1_CASE_MAINTENANCE_CANDIDATES.md#2-coverage-to-reachability-ratio) for self-exclusion, evaluation of each case's own problem, and zero-denominator handling.

\[
CM(c)=\frac{|\operatorname{Cov}(c)|}{|\operatorname{Reach}(c)|}.
\]

**Reading the formula.** The numerator counts problems \(c\) solves. The denominator counts cases solving \(c\)'s own problem. This presumes that the problem corresponding to \(c\) is represented in the competence model.

**Example and scale.** Covering 6 problems while having 2 solvers for its own problem gives \(CM(c)=3\). The ratio is nonnegative when defined but is not bounded by 1. Unlike RC, it does not separately count alternatives for each covered problem. Adding an epsilon to handle a zero denominator would modify the displayed formula.

This rewards cases that solve many problems while being replaceable by few cases [S6]. The equation was checked in S1, Equation 4, and S13, Equation 9; the original publisher abstract was accessible, but its full text was not verified here.

**Our interpretation:** the denominator concerns alternatives for `c` itself, unlike RC, which considers alternatives for every problem `c` solves. These are different redundancy proxies. An implementation must specify self-coverage and zero-denominator handling.

### 5. Harm and protected coverage: Delany and Cunningham (2004)

Blame-Based Noise Reduction (BBNR) records a liability set: queries misclassified when `c` was retrieved with an opposing label. It considers high-liability cases for deletion, but checks that their previously covered cases can still be classified correctly without them. Conservative Redundancy Reduction (CRR) uses coverage ordering to favor retaining boundary cases [S5, Sections 3.1-3.3].

Its sets can be written explicitly in this note's notation:

\[
\operatorname{Cov}(c)=
\{q\in D:c\in N_k(q),\ y_c=y_q,\ f_S(q)=y_q\},
\]
\[
\operatorname{Liability}(c)=
\{q\in D:c\in N_k(q),\ y_c\ne y_q,\ f_S(q)\ne y_q\}.
\]

All three conditions on each line must hold. A disagreeing neighbor is not assigned liability when the aggregate classifier still answers correctly. An agreeing neighbor is not assigned coverage when the aggregate classifier answers incorrectly. These distinctions connect directly to the reputation score in Section 10.

A compact restatement of BBNR's deletion test is:

\[
|\operatorname{Liability}(c)|>0
\quad\text{and}\quad
\forall q\in\operatorname{Cov}(c),\; f_{S\setminus\{c\}}(q)=y_q.
\]

This is a decision rule, not a weighted score. For multiple neighbors, participation in a wrong vote is not proof that the individual case caused the error. The removal check adds evidence. Coverage preservation here is defined on the checked cases, not every future query.

**Example.** Suppose \(c\) has liability on 3 queries and coverage on 2. Temporarily remove it and reclassify both covered queries. If either becomes wrong, restore \(c\); if both remain correct, deletion passes this test. Passing does not itself show that the 3 previously wrong answers improve.

**CRR's different ordering.** CRR stands for Conservative Redundancy Reduction. It presents small-coverage cases first for retention, then removes the cases they cover from the candidate list. Small coverage is used as a boundary-case heuristic. Thus "more coverage means retain earlier" is not a rule shared by all coverage-based algorithms.

### 6. Relative adaptation performance: Leake and Wilson (2000)

The method scores the adaptation cost saved relative to alternative cases:

\[
RP(c)=\sum_{q\in\operatorname{Cov}(c)}
\left[1-\frac{\operatorname{AdaptCost}(c,q)}
{\max_{j\in\operatorname{Reach}(q)\setminus\{c\}}\operatorname{AdaptCost}(j,q)}\right].
\]

The denominator is the **worst alternative**, not the best. The authors also discuss weighting problems by expected frequency. RP-CNN uses an initial ranking to avoid repeated expensive computation [S7, Section 5].

**Reading the formula.** \(\operatorname{AdaptCost}(c,q)\) is the computational effort required to adapt \(c\)'s solution to \(q\). The maximum ranges over other cases that can solve \(q\). Each summand is the fraction of that worst alternative's cost saved by using \(c\). Both costs must be in the same units.

**Example and sign.** If \(c\) costs 2 and alternatives cost 3 and 10, its contribution is \(1-2/10=0.8\). If \(c\) instead costs 12, it is \(1-12/10=-0.2\). With nonnegative costs and a positive denominator, each summand is at most 1 but can be negative. Consequently RP is neither a probability nor necessarily nonnegative. More use cases can increase its total magnitude.

**Connection to RC.** Both sum over covered problems. RC discounts credit by the number of solvers; RP uses their adaptation costs. Two cases can have identical RC but different RP.

**Our interpretation:** this adds information that binary coverage omits: two memories may solve the same problems with different effort. A neural correction magnitude is not automatically equivalent to this paper's computational adaptation cost. Empty alternative sets and zero costs require explicit treatment before reuse of the formula.

### 7. Adaptation knowledge value: Jalali and Leake (2014)

A case can be useful as a retrieved source and as material for generating adaptation rules. AGCBM1 scores both roles using reciprocal prediction errors from leave-one-out trials:

\[
\operatorname{CaseComp}(c)=\sum_{j\in\mathcal U_c}\frac{1}{e_j+\epsilon},\qquad
\operatorname{AdaptComp}(c)=\sum_{j\in\mathcal A_c}\frac{1}{e_j+\epsilon},
\]
\[
\operatorname{Comp}(c)=\alpha\operatorname{CaseComp}(c)+(1-\alpha)\operatorname{AdaptComp}(c).
\]

**Reading the symbols.**

- \(\mathcal U_c\) contains prediction events in which \(c\) was used as a retrieved source.
- \(\mathcal A_c\) contains adaptation-use events involving rules derived from \(c\). A case-difference rule is obtained by comparing two stored cases and relating differences in their problems to differences in their solutions.
- \(e_j\ge0\) is the estimation error of the relevant prediction. It is not a time cost or a difference between predictions with and without \(c\).
- \(\epsilon>0\) prevents division by zero. A perfect prediction contributes \(1/\epsilon\).
- \(\alpha\in[0,1]\) sets the balance: 1 uses only source-case competence; 0 uses only adaptation competence.

**Example and scale.** With source-use errors \(0.1,0.4\) and \(\epsilon=0.1\), \(\operatorname{CaseComp}=5+2=7\). If the adaptation contribution is 3 and \(\alpha=0.6\), the combined score is \(0.6(7)+0.4(3)=5.4\). These sums are nonnegative and exposure-dependent, not probabilities. One prediction can credit several cases or rules; the score does not conserve a fixed total credit.

Here `U_c` indexes source-case uses and `A_c` indexes uses of rules derived from the case. Unused roles receive zero. These scores guide case-base construction [S8, Equations 1-3].

**Our interpretation:** the two-role principle is relevant to neural reuse. The exact reciprocal-error formula is sensitive to exposure and `epsilon`; it is not a measured causal training influence. The printed Algorithm 1's error-threshold condition should be reconciled with the CNN narrative before reproducing that selector.

### 8. Jointly necessary cases: Mathew and Chakraborti (2017)

FootprintCA represents problems solved by combinations of cases. Its recursive retention score accounts for covered cases and the supporting cases required alongside a candidate [S9, Equation 1]:

An **AND relation** means all cases in a group are required. An **OR relation** means alternative groups can solve the same problem. For example, \((a\ \mathrm{AND}\ b)\ \mathrm{OR}\ d\) means that either the pair \(\{a,b\}\) or the single case \(d\) solves \(q\). Neither \(a\) nor \(b\) alone suffices.

Here \(\operatorname{CoveredCases}(c)\) includes problems solved by \(c\) alone **or with support**. This is broader than single-case coverage. \(\operatorname{SupportCases}(c,q)\) contains the other cases required alongside \(c\) in the represented solution of \(q\). In the example, \(\operatorname{SupportCases}(a,q)=\{b\}\), while \(d\) needs no support.

\[
RS_{t+1}(c)=\sum_{q\in\operatorname{CoveredCases}(c)}
\frac{RS_t(q)}{1+\sum_{j\in\operatorname{SupportCases}(c,q)}RS_t(j)}.
\]

**Reading the recurrence.** \(RS_t\) is the score vector at iteration \(t\). Each covered problem contributes its current score, discounted by the current scores of the required supporting cases. The added 1 makes the denominator valid when no support is required. All right-hand scores come from the previous iteration. These are recursively computed graph scores, not probabilities or neural activations.

**Initialization.** The paper also defines the starting scores [S9, Equation 2]. Let \(\mathcal G_{-c}(q)\) denote the alternative solving case sets that exclude \(c\), as counted by its competence representation. In equivalent notation:

\[
RS_0(c)=\sum_{q\in\operatorname{CoveredCases}(c)}
\frac{1}
{(1+\lvert\mathcal G_{-c}(q)\rvert)
 (1+\lvert\operatorname{SupportCases}(c,q)\rvert)}.
\]

This discounts both alternative solutions and required partners. For one covered problem with one alternative solving set and one required partner, the initial contribution is \(1/(2\cdot2)=1/4\). Later updates use the partners' scores, not simply their number. Reproduction requires a defined solution graph, treatment of multiple solution routes, and numerical iteration/stopping convention; the displayed recurrence alone is not a complete implementation.

Initialization also considers alternative solutions. The score orders a footprint-building procedure that checks whether retained subsets solve each case. This equation is specifically from the IJCAI abridged paper; do not silently mix it with the different formulation in the full journal paper or S13.

**Our interpretation:** NN-kNN's combined case contributions make interaction effects relevant. However, a soft weighted prediction is not automatically the paper's explicit AND-OR solving graph. Constructing that graph and computing the recurrence are additional assumptions and costs.

### 9. Changing workloads: Lu et al. (2016)

NEFCS combines drift detection with noise handling and reversible activation decisions. It tracks recent predictive performance with confidence intervals. Low upper-bound accuracy can deactivate a case; improved lower-bound accuracy can reactivate it. Drift detection helps avoid treating a new concept as noise [S10, Section 3.3].

NEFCS expands to **Noise-Enhanced Fast Context Switch**. A context here is a pattern of problem-solution relationships currently relevant to the stream. Deactivation makes a stored case unavailable for ordinary reasoning while retaining the possibility of reactivation.

**The confidence interval.** The paper keeps recent classification records per case. In standardized notation, let \(n_c>0\) be the number of recorded attempts and \(\widehat p_c\) their observed success fraction. Its interval is [S10, Equation 1]:

\[
[L_c,U_c]=
\frac{\widehat p_c+z^2/(2n_c)
\ \mathord{\pm}\ z\sqrt{\widehat p_c(1-\widehat p_c)/n_c+z^2/(4n_c^2)}}
{1+z^2/n_c}.
\]

The minus sign gives \(L_c\); the plus sign gives \(U_c\). The coefficient \(z\) sets the confidence level. Fewer observations generally leave greater uncertainty. Both bounds concern estimated predictive success, not confidence that the case's content is intrinsically true.

The paper deactivates when \(U_c<p_{\max}\) and reactivates when \(L_c>p_{\min}\). These threshold names follow the paper: \(p_{\max}\) is the inactivation threshold and \(p_{\min}\) is the acceptance threshold. They are policy settings, not extra terms in a scalar score. The interval is undefined at zero observations; an evidence-initialization rule is needed.

SRR orders candidates by reachability and removes one only when it and previously removed cases linked to it remain solvable. It tracks preserved and temporarily locked cases during successive removal rounds [S10, Section 3.4].

SRR means **Stepwise Redundancy Removal**. Its bookkeeping matters because an earlier deletion can make a later candidate necessary. It checks support for already removed cases as well as the current candidate; it does not repeatedly discard cases using an unchanged redundancy ranking alone.

**Our interpretation:** this contributes evidence sufficiency, time sensitivity, recovery, and sequential coverage bookkeeping. It is not merely a recency term to add to a static score. Case age alone does not determine usefulness.

### 10. Reputation: Nakhjiri et al. (2020) — source S11

Reputation increases or decreases whenever a case is used as a neighbor in leave-one-out classification. In equivalent summation notation:

\[
\operatorname{Rep}(c)=\sum_{q:\,c\in N_k(q)}
\big(\mathbf1[y_c=y_q]-\mathbf1[y_c\ne y_q]\big).
\]

**Reading the formula.** The condition under the sum selects queries whose nearest-neighbor sets contain \(c\). This is a **reverse-neighbor set**: queries pointing toward \(c\), rather than \(c\)'s own nearest neighbors. Every agreeing use contributes \(+1\); every disagreeing use contributes \(-1\). The indicator tests the stored class labels, not whether the entire classifier was right.

**Example and exposure.** Seven agreeing uses and three disagreeing uses give \(7-3=4\). With \(n_c\) total uses, the range is \([-n_c,n_c]\). For \(n_c>0\), let \(\widehat p_c\) be its agreement fraction. Algebraically,

\[
\operatorname{Rep}(c)=n_c(2\widehat p_c-1).
\]

This is an equivalent rewriting, not a new score. It shows the mixture of agreement quality and exposure count. Two cases with the same agreement rate can receive very different reputations. The \(\widehat p_c\) here measures label agreement; it should not silently replace the success statistic in NEFCS.

The sum is over queries that retrieve `c`, not necessarily the cases in `c`'s own nearest-neighbor list. Updates occur regardless of whether the aggregate prediction is correct. RBM variants use reputation thresholds and additional procedures [S11, Definition 1, Algorithm 1].

**Our interpretation:** this is a well-defined, inexpensive classification baseline. A zero can mean no exposure or balanced positive and negative evidence. The formula does not directly test replacement, unique coverage, or continuous-target utility. Those omissions matter even if its overall benchmark accuracy is good.

### 11. Reliability with unreliable neighbors: Parsodkar et al. (2022) — source S12

**Paper and purpose.** S12 is *Never Judge a Case by Its (Unreliable) Neighbors: Estimating Case Reliability for CBR*. Its method is called **RelCBR**. It estimates which stored problem-solution pairs deserve more trust. The authors use the estimates to give unreliable cases less weight during prediction and to direct expert review toward those cases [S12, Sections 2, 4, and 6].

**The problem it addresses.** A common way to judge a case is to compare its solution with the solutions of nearby cases. But a correct case can disagree with its neighbors because those neighbors contain errors. RelCBR therefore lets some neighbors count more than others. The difficulty is that their reliability is also unknown.

The paper resolves this by estimating all case reliabilities together. Its working assumption is that reliable cases should be predictable from other reliable cases with similar problems. The paper calls this a **circular definition**: the reliability assigned to one case affects how the method judges other cases, which in turn affects the first case. Operationally, this means repeatedly updating a vector of reliability values to reduce a specified loss. It does not require a trusted case to be identified in advance.

**What each symbol means.** We use the survey's notation below. The paper writes \(v_c\) for the stored solution and \(v_c^e\) for its estimate; these correspond to \(y_c\) and \(\widehat y_c\) here.

| Symbol | Meaning |
|---|---|
| \(C\) | The case base being assessed. This is a set, distinct from our per-case statistic \(C_i\). |
| \(c\) | The case currently being assessed. Its stored solution may be wrong. |
| \(N_k(c)\) | Its \(k\) nearest neighbors according to similarity between problem descriptions. Case \(c\) itself is excluded. |
| \(s^P(c,j)\) | Similarity between the problem descriptions of \(c\) and neighbor \(j\). A larger value means more similar problems. |
| \(y_j\) | Neighbor \(j\)'s stored solution: a number for regression or a class label for classification. |
| \(r_j\in[0,1]\) | The reliability value being fitted for case \(j\). It is reused whenever \(j\) participates in a prediction. |
| \(\boldsymbol r\) | The vector containing the reliability values of all cases. |
| \(\widehat y_c(\boldsymbol r)\) | The solution predicted for \(c\) using its neighbors and the current reliability estimates. |

The optimization holds the chosen similarities and neighborhoods fixed while estimating \(\boldsymbol r\). Each case takes two roles: it is a problem to be predicted from its neighbors, and it can be a neighbor used to predict other cases. Predicting each case without using that case itself is called **leave-one-out prediction** [S12, Sections 4.1-4.3].

**Step 1: predict each case from its neighbors.** A neighbor contributes in proportion to both its similarity and its reliability. Divide by the sum of these products so that the prediction is a weighted average:

\[
\widehat y_c(\boldsymbol r)=
\frac{\sum_{j\in N_k(c)}r_j\,s^P(c,j)\,y_j}
{\sum_{j\in N_k(c)}r_j\,s^P(c,j)}.
\]

This is Equation (3) in the paper. The stored solution \(y_c\) and reliability \(r_c\) of the case being predicted do not enter this prediction. They enter its loss in Step 2.

**Worked example, using illustrative reliability values.** Suppose a case has stored solution 11. Its three neighbors have equal similarity to it:

| Neighbor | Stored solution | Current reliability | Reliability times solution |
|---|---:|---:|---:|
| A | 10 | 0.9 | 9.0 |
| B | 12 | 0.9 | 10.8 |
| C | 30 | 0.1 | 3.0 |

Equal weighting would predict \((10+12+30)/3\approx17.33\). With the displayed reliabilities, the prediction is

\[
\widehat y_c=\frac{9+10.8+3}{0.9+0.9+0.1}=12.
\]

The lower reliability of neighbor C reduces its effect. The values 0.9, 0.9, and 0.1 are assumed here to illustrate prediction. The next step explains how RelCBR estimates them.

**Step 2: choose reliabilities that reduce prediction error.** For a numeric solution, the error for case \(c\) is \(y_c-\widehat y_c\). RelCBR squares this error, multiplies it by the reliability of \(c\), and adds the results across cases:

\[
\min_{\boldsymbol r}L(C,\boldsymbol r),\qquad
L(C,\boldsymbol r)=\sum_{c\in C}r_c\big(y_c-\widehat y_c(\boldsymbol r)\big)^2,
\qquad 0\le r_c\le1,\quad \sum_{c\in C}r_c\ge a>0.
\]

The loss is Equation (6); the constraints are Equations (8)-(9). Reliability has **two roles**:

1. **Trust in a case's stored solution.** The factor \(r_c\) outside the squared error determines how strongly the method tries to reproduce \(y_c\). If that stored solution is unreliable, the method can reduce its importance by lowering \(r_c\).
2. **Influence when predicting other cases.** Inside other cases' predictions, \(r_c\) determines how much case \(c\)'s solution contributes. Changing \(r_c\) can therefore improve or worsen many other predictions.

This is why the method estimates all reliabilities together. It does not simply set each reliability from that case's own prediction error. The numeric gradient in Equation (11) explicitly includes both its own error and its effect on the errors of cases that use it as a neighbor.

The constraint \(\sum_c r_c\ge a\) prevents the method from avoiding all errors by assigning essentially no reliability to any case. Here \(a\) is a positive, user-selected minimum total reliability, unrelated to our activation notation \(a_i(q)\). Since every \(r_c\le1\), a feasible choice requires \(a\le|C|\). It does not specify a number of cases to delete or retain.

The reliability values themselves **do not have to sum to one**. It is the prediction weights that are normalized. For a particular case \(c\), the share assigned to neighbor \(j\) is

\[
w_{cj}=\frac{r_j s^P(c,j)}{\sum_{\ell\in N_k(c)}r_\ell s^P(c,\ell)},
\qquad \sum_{j\in N_k(c)}w_{cj}=1.
\]

Here \(w_{cj}\) is just the normalized weight appearing in Step 1. Nonnegative similarities and a positive denominator are required. A positive total reliability over the entire case base does not ensure a positive denominator in every neighborhood; an implementation must handle that local condition.

**Step 3: update the reliabilities repeatedly.** The paper's Algorithms 1-2 start each reliability at 0.5. Each iteration predicts the cases from their neighbors, calculates how changes to the reliabilities would change the total loss, and moves the reliabilities in a direction that reduces the loss. It then brings the values back within the two constraints above. This procedure is called **projected gradient descent**. The iteration stops when the largest change in any reliability is no greater than a chosen tolerance \(\epsilon\). The learning rate controls the size of each update [S12, Equations 10-13 and Algorithms 1-2].

**How classification differs.** Instead of averaging numeric solutions, the model adds normalized neighbor weights separately for each class. This gives a vector of class probabilities. For example, the prediction \((0.8,0.2)\) assigns probabilities 0.8 and 0.2 to two classes. If the case's stored class is the first one, represent its label as the one-hot vector \((1,0)\). The squared error is

\[
(1-0.8)^2+(0-0.2)^2=0.08.
\]

The loss contribution is \(0.08r_c\). More generally it is \(r_c\|\boldsymbol e_{y_c}-\widehat{\boldsymbol p}_c\|_2^2\), where \(\boldsymbol e_{y_c}\) has 1 at the stored class and 0 elsewhere, \(\widehat{\boldsymbol p}_c\) is the predicted probability vector, and the squared norm means summing squared differences across classes. Prediction selects the class with the largest probability, but reliability fitting uses these probability errors rather than only a correct/incorrect indicator [S12, Equations 4-7].

**What the fitted score means.** A higher \(r_c\) means the case receives more trust under this joint prediction model. A low value makes it a candidate for review and reduces its influence in predictions. It is not a calibrated probability that the stored solution is true. The procedure uses the potentially incorrect stored solutions themselves when fitting reliabilities; a corrected case base is available for evaluation in the experiments, not supplied as the target answers during reliability estimation. A group of mutually consistent errors or a valid case near a class boundary can still be difficult to assess.

**What was demonstrated.** The paper tests synthetic numeric and categorical case bases with known corrections, plus Boston Housing and Iris with deliberately introduced errors. In its synthetic housing experiment, Table 4 reports MSE 67.32 for RelCBR versus 225.71 for ordinary CBR. In Iris, 20 labels are randomly changed per run; over 100 runs, Table 8 reports mean error 0.0297 for RelCBR versus 0.0451 for ordinary CBR. These are the paper's comparisons under its leave-one-out protocols. They do not establish performance for learned NN-kNN retrieval or neural adaptation [S12, Section 5].

**Connection to our C/H rule.** The following comparison is our interpretation:

| Question | RelCBR | Our accepted C/H rule when adaptation is enabled |
|---|---|---|
| Where does evidence come from? | Predict each stored case from its neighbors and compare with its potentially incorrect stored solution. | Observe the final adapted outcome on evaluated queries that retrieve the case. |
| What is calculated? | A reliability \(r_i\in[0,1]\), fitted jointly with all other reliabilities by minimizing a loss. | Cumulative helpful and harmful outcomes, weighted by the case's activation. |
| How does a case affect other cases' evaluation? | Its reliability changes their predictions; its value is optimized with those effects included. | Participating cases share each query's success or failure according to their activations. |
| What is normalized per query? | Similarity times reliability, after division by the neighborhood total. | The case activations used by NN-kNN. |
| How is reuse represented? | Reliability-weighted combination of stored solutions. The paper does not define a neural correction network. | The evaluated outcome includes neural adaptation when enabled. |
| Does this alone select the active case base? | No. The paper proposes reliability weighting and review prioritization. | No. The comprehensive case maintenance score and selection rule remain under development. |

**Implication for the general maintenance formula.** RelCBR gives us a concrete example of estimating case quality through a prediction objective. Its reliability affects both the prediction and the importance assigned to each stored target. To recover RelCBR as a special case, a general method would need to reproduce these two roles, the reliability-weighted neighbor prediction, and the constraints. Setting \(r_i=C_i/(C_i+H_i)\) would not by itself reproduce its optimization. This is a possible connection to develop, not an established reduction or an adopted extra component.

### 12. EnergyCompress: Badra et al. (2025) — source S15

**The main idea.** EnergyCompress asks how much prediction would worsen or improve if a particular case were removed. It scores every case this way, removes the case with the lowest score, and repeats. The goal is to find a smaller case base that performs well for the chosen predictor. The paper is *EnergyCompress: A General Case Base Learning Strategy*, IJCAI 2025 [S15].

The score concerns a case's value to the current prediction system. A factually correct case may be redundant. A case may also be harmful for one predictor and useful for another. The paper calls the measured value **case competence**. It defines it through three steps: assign prediction energies, calculate prediction loss, and measure how the loss changes after case removal [S15, Sections 3.1-3.3].

**Notation used below.** These symbols restate the paper's equations using the survey's notation.

| Symbol | Meaning |
|---|---|
| \(S\) | The current case base used for prediction. The paper writes \(CB\). |
| \(c\in S\) | The stored case whose value is being measured. |
| \(D\) | A fixed set of reference problems with known answers, used to evaluate predictions. The paper writes \(T_{ref}\). |
| \((x,y)\in D\) | One reference problem \(x\) and its correct class label \(y\). |
| \(E_S(x,y)\) | The energy assigned to candidate answer \(y\) by the predictor using \(S\). Lower is better. |
| \(\ell_S(x,y)\) | The loss for one reference problem. Lower is better. |
| \(L_D(S)\) | Mean loss across the reference problems. |
| \(\lambda\) | The desired gap between the correct answer's energy and the best competing answer's energy. |

Keep \(D\) fixed while shrinking \(S\). A query remains an evaluation problem even if its corresponding stored case is removed. Otherwise, deleting a difficult case could also delete the evidence that the system performs poorly on it.

**1. Energy defines a prediction preference.** \(E_S(x,y)\) is a scalar measuring the incompatibility of candidate answer \(y\) with input \(x\), given \(S\) and the fixed prediction mechanism. Lower energy is preferred. Prediction chooses:

\[
f_S(x)=\arg\min_{y\in\mathcal Y}E_S(x,y),
\]

where \(\mathcal Y\) is the set of possible class labels. The symbol \(\arg\min\) means choosing the label with the smallest energy. Energy is a numerical prediction score here; it does not mean computation time or electricity use. Table 1 supplies different energies for different predictors. For a predictor that outputs class probabilities, one choice is \(E_S(x,y)=1-P_S(y\mid x)\), where \(P_S(y\mid x)\) is its predicted probability for class \(y\). A probability of 0.8 then gives energy 0.2. This construction lets an existing probabilistic predictor supply the energies without requiring a new neural energy model.

**2. Hinge loss measures whether the correct answer wins by enough.** The **margin** \(\lambda\) is the desired gap: the correct answer should have energy at least \(\lambda\) lower than every wrong answer. The minimum over wrong answers finds the strongest competitor. The **hinge loss** is zero when the required gap is achieved; otherwise it measures the shortfall. Thus it can penalize a prediction that is correct but only narrowly ahead:

\[
\ell_S(x,y)=\max\left(0,\lambda+E_S(x,y)-\min_{y'\ne y}E_S(x,y')\right),
\qquad L_D(S)=\frac1{|D|}\sum_{(x,y)\in D}\ell_S(x,y),
\]
\[
\operatorname{Competence}(c\mid S,D)=L_D(S\setminus\{c\})-L_D(S).
\]

Here \(y'\) ranges over incorrect labels, and \(|D|\) is the number of reference problems. The first line defines prediction loss and its average. The second line measures the change in that average when case \(c\) is removed; Step 3 explains its sign.

**Probability example.** With \(E=1-P\), substitution gives
\[
\ell_S(x,y)=\max\bigl(0,\lambda+\max_{y'\ne y}P_S(y'\mid x)-P_S(y\mid x)\bigr).
\]
For \(\lambda=0.2\), these illustrative predictions give:

| Correct-class probability | Largest wrong-class probability | Hinge loss | Interpretation |
|---:|---:|---:|---|
| 0.70 | 0.20 | 0 | Correct with a probability gap of 0.50, exceeding the required 0.20. |
| 0.40 | 0.35 | 0.15 | Correct, but the gap is only 0.05. The shortfall is 0.15. |
| 0.30 | 0.50 | 0.40 | Incorrect; the strongest wrong class leads by 0.20. |

The unused probability in each row belongs to other classes. The margin here is an explanatory choice, not an adopted NN-kNN setting.

**3. Remove a case and measure the change in loss.** The whole case base's competence is \(-L_D(S)\): a smaller loss gives a higher competence, with zero as the largest possible value. The competence of an individual case is the drop in the base's competence after removing it. Substituting the negative-loss definition gives the equation above: loss without the case minus loss with it.

Suppose the current mean loss is 0.10. Temporarily remove each candidate separately and predict the same reference problems again:

| Candidate removed | Mean loss after removal | Case competence | Meaning |
|---|---:|---:|---|
| A | 0.16 | +0.06 | Removing A worsens the predictions under this loss. |
| B | 0.10 | 0 | Removing B leaves mean loss unchanged. |
| C | 0.07 | -0.03 | Removing C improves the predictions under this loss. |

EnergyCompress removes C first because it has the lowest competence. These are illustrative losses, not experimental results. Each candidate is compared with the same current case base. After C is actually removed, the baseline changes to 0.07, and the remaining cases must be assessed again.

**4. Repeat deletion and remember the best case base.** Algorithm 1 proceeds as follows:

1. Measure the current case base's accuracy on \(D\) and keep a copy if it is the best seen so far.
2. Score each remaining case by its effect on mean hinge loss.
3. Remove the case with the lowest score.
4. Recompute for the smaller case base and repeat.
5. Return the visited case base with the highest reference accuracy. If accuracies tie, retain the later, smaller case base.

The algorithm continues through progressively smaller case bases. It does not stop merely because all current removal scores become positive. It also does not examine every possible subset, so the returned set is the best along its deletion sequence, not a demonstrated global optimum.

**Scale and selector.** Energy scale and the chosen margin affect the score. Energies that give identical class predictions can still produce different loss differences. The paper uses hinge loss to choose the next deletion but accuracy to select the final visited set; these objectives are not identical. A score of zero can also reflect hinge-loss saturation, so it need not establish that a case has no value on other workloads.

**Sign convention.** Positive case competence means removal hurts; negative means removal helps. The paper separately names a per-query quantity `influence` and defines it as loss with the case minus loss without it. Its sign is therefore the opposite of the case-competence score used for deletion. Preserve this distinction when comparing equations or implementing the method.

Experiments use fixed similarity and classification. Regression and interaction with similarity learning are left for future work. The published stopping/selection policy does not enforce a user-selected `K` [S15, Algorithm 1 and conclusion].

**Connection to combined coverage and redundancy.** This is our interpretation. If another retained case can replace a case without increasing reference loss, removal can have a small score. If removing the case leaves an important reference problem poorly solved, its score increases. Thus the measurement considers alternatives already present in the case base. This does not guarantee protection of every rare class: loss is averaged over \(D\), so the result depends on which problems are represented and how often they occur.

**Connection to our C/H rule.** Our accepted rule gives every participating case an activation-weighted share of the adapted query's success or failure. EnergyCompress evaluates a second prediction with a case removed and measures the difference. For example, a case can receive a positive C update on a successful query even if removing it would leave the prediction unchanged. These methods measure different aspects of case usefulness. C/H can be updated from ordinary evaluated queries; removal comparisons require additional predictions.

For NN-kNN, a corresponding removal comparison would keep the chosen model checkpoint fixed, remove the case, recompute retrieval and normalized activations, and run reuse again. When adaptation is enabled, evaluate the final adapted outputs in both conditions. Merely subtracting the removed case's previous weighted contribution would not reproduce this comparison. This is an NN-kNN extension to investigate, not something validated in the paper.

**PI refinement, 2026-09-20:** Adopt this as a candidate comparison using final prediction loss only; report the adaptation penalty separately. Preserve parameters trained using removed cases. Cache queries where each case's activation exceeds a threshold to reduce reevaluation cost. This approximates full-query loss change when omitted queries can still be affected. Optional retraining is a higher-cost branch. See [the removal protocol and equations](../03-research-architecture/T1_CASE_MAINTENANCE_CANDIDATES.md#3-influence-measured-by-case-removal).

**What must match to recover EnergyCompress.** A general case maintenance score could use this difference in losses as a special case. Exact recovery requires the same energy, hinge loss, margin, reference set, and prediction behavior after removal. Recovering the complete method additionally requires its repeated deletion and final accuracy-based selection. Using another task loss or imposing our fixed case budget would produce a modified method. The cost of the full sequence includes repeated case scoring; it is not just one scoring pass.

### 13. Fidelity versus competence: Parsodkar et al. (2024)

This survey separates intrinsic problem-solution quality from a case's importance to the current reasoner's competence. It argues that the two should interact but cannot be inferred from one another [S13, Sections 2 and 4]. It also catalogs several inexpensive neighborhood metrics:

These metrics compare **problem-space neighborhoods**, selected using input descriptions, with labels or **solution-space neighborhoods**, selected using stored solutions. The same case identifiers can therefore appear in two different neighbor lists.

**Cohesion: overlap between two neighbor lists.** Let \(N^P(c)\) and \(N^Y(c)\) be the problem- and solution-space neighbor sets. Cohesion is their Jaccard similarity:

\[
\operatorname{Cohesion}(c)=
\frac{\lvert N^P(c)\cap N^Y(c)\rvert}
{\lvert N^P(c)\cup N^Y(c)\rvert}.
\]

When the union is nonempty, the range is \([0,1]\); larger means more overlap. For sets \(\{a,b,c_1\}\) and \(\{b,c_1,d\}\), it is \(2/4=0.5\). This tests agreement of neighborhood membership, not how much the target case improves predictions. Also, similar solutions need not imply similar problems, which can make the symmetric overlap test too restrictive.

**Alignment: solution agreement weighted by problem similarity.** Using problem-space neighbors \(N^P(c)\),

\[
\operatorname{Alignment}(c)=
\frac{\sum_{j\in N^P(c)}s^P(c,j)s^Y(c,j)}
{\sum_{j\in N^P(c)}s^P(c,j)}.
\]

This is a weighted average of solution similarities. If problem weights are nonnegative, their sum is positive, and solution similarities lie in \([0,1]\), alignment also lies in \([0,1]\). With classification similarity \(s^Y(c,j)=\mathbf1[y_c=y_j]\), it becomes weighted label agreement. Weights \(0.8,0.2\) on one agreeing and one disagreeing neighbor give alignment \(0.8\).

**Complexity: disagreement over increasing neighborhood sizes.** Let \(m\) be the largest size examined and
\[
p_{y_c}(N_k(c))=\frac{1}{k}\sum_{j\in N_k(c)}\mathbf1[y_j=y_c].
\]
The complexity is
\[
\operatorname{Complexity}(c)=
1-\frac1m\sum_{k=1}^{m}p_{y_c}(N_k(c)).
\]

It averages agreement fractions across nested neighborhoods, then takes their complement. The range is \([0,1]\), assuming each requested neighborhood exists. Higher means more disagreement. For \(m=2\), with the nearest neighbor agreeing and the second disagreeing, the fractions are \(1\) and \(1/2\), so complexity is \(0.25\). Early neighbors are counted at several neighborhood sizes; this differs from simply measuring disagreement among the \(m\) nearest neighbors once.

**Friend-to-enemy ratio: relative distances.** Let \(N_k^+(c)\) contain the \(k\) nearest same-class cases and \(N_k^-(c)\) the \(k\) nearest different-class cases:
\[
FE(c)=
\frac{\sum_{j\in N_k^+(c)}d(c,j)}
{\sum_{j\in N_k^-(c)}d(c,j)}.
\]

With both sets of size \(k\), this is also the ratio of their mean distances. Smaller values mean same-class neighbors are closer relative to other-class neighbors. A numerator of 2 and denominator of 8 give \(0.25\); reversing them gives 4. It is nonnegative when defined but has no fixed upper bound.

**Connection and limitation.** Cohesion, alignment, complexity, and FE inspect the neighborhood **of** the case. RBM instead aggregates queries that retrieve that case. A valid rare-class or boundary case can look inconsistent with its neighbors. Neighborhood disagreement is evidence to examine, not proof of an incorrect label.

These equations were checked in S13, Equations 1-3 and 6, not independently rederived from every original paper. Empty neighborhoods, insufficient members of either class, and zero denominators need explicit conventions. These are proxies for fidelity or local consistency, not complete selectors under a memory budget.

### 14. Beyond whole-case deletion: Leake and Schack (2015)

Flexible Feature Deletion removes selected portions of individual cases. This matters when cases differ in size or remain useful at reduced detail. It exposes a limitation of case-count-only compression: equal numbers of cases need not occupy equal memory [S14].

**Reading the operation.** Feature deletion here can remove selected attributes from an individual case. It does not necessarily remove that attribute from every case. The retained objects can therefore differ in detail and storage size.

For comparison, a case-count budget is \(\lvert S\rvert\le K\). A byte budget would be \(\sum_{c\in S}b(c)\le B\), where \(b(c)\) is stored bytes per case and \(B\) is available bytes. This second inequality is an explanatory resource-accounting distinction, not an adopted replacement budget or a reproduction of S14's optimizer.

**Our interpretation:** relevant to later heterogeneous memories, but it does not change the PI-approved initial fixed-`K` formulation. Report byte memory and runtime as well as count. Treat partial-case compression as an extension, not an already accepted T1 mechanism.

## Comparing and connecting the methods

This section is an explanatory synthesis. Equal symbols make the differences easier to see; they do not establish equivalence between methods.

### What is measured, and in which direction?

| Measure | What is counted or evaluated? | Direction and scale | What a low score does not establish |
|---|---|---|---|
| Utility \(U\) | Expected computational savings minus overhead | Larger is better; signed cost units | That the case lacks unique capability |
| Relative coverage \(RC\) | Shared credit across covered problems | Larger gives earlier consideration; nonnegative count-like sum | That all low-ranked cases can safely be removed together |
| Coverage ratio \(CM\) | Coverage count divided by reachability of the case itself | Larger is favored; nonnegative, not bounded by 1 | That alternatives exist for every problem it covers |
| BBNR liability | Wrong predictions involving a disagreeing retrieved case | Larger is more suspicious; nonnegative count | That an unused or redundant case is valuable |
| Relative performance \(RP\) | Adaptation savings relative to worst alternatives | Larger is better; signed sum of cost ratios | That a case has low prediction accuracy |
| AGCBM competence | Reciprocal errors credited to source and adaptation-rule uses | Larger is better; nonnegative, exposure-dependent | That a seldom-used case is unreliable |
| FootprintCA retention score | Covered-problem value, alternatives, and required partners | Larger gives earlier consideration; recursive score | That case interactions can be ignored |
| NEFCS interval | Recent predictive success and its uncertainty | Higher success is favorable; two bounds in \([0,1]\) | That limited evidence proves permanent obsolescence |
| RBM reputation | Agreement minus disagreement over reverse-neighbor exposure | Larger is favored; signed count | Whether zero means no exposure or balanced evidence |
| RelCBR reliability | Jointly fitted local consistency under reliability weighting | Larger means more reliable under its model; \([0,1]\) | That a low-reliability case has no unique task coverage |
| EnergyCompress case competence | Increase in mean hinge loss after removal | Larger means removal hurts more; signed loss difference | That zero means no value beyond this set, margin, and workload |
| Cohesion / alignment | Neighborhood overlap / weighted solution agreement | Larger agreement; conditionally \([0,1]\) | That a valid boundary or rare case should be deleted |
| Complexity / FE ratio | Multiscale disagreement / relative friend-enemy distances | Smaller suggests greater local consistency; \([0,1]\) / unbounded nonnegative ratio | That large values certify a wrong label |

The scales are not commensurate. A reputation of 5, RC of 5, and adaptation competence of 5 have different meanings. Even scores in \([0,1]\) can measure different objects. Normalizing ranges would not by itself make their evidence independent or comparable.

### A common coverage example

Assume four equally weighted reference problems and the following **fixed single-case solving relation**. A problem is solved whenever at least one of its listed solvers remains. This simplifying assumption makes coverage monotone: adding a solver cannot hurt. It is not generally true of a classifier with interacting votes.

| Case | Problems it solves | Coverage count | Relative coverage | Problems lost if this case alone is removed from \(\{a,b,c\}\) |
|---|---|---|---|---|
| \(a\) | \(q_1,q_2,q_3\) | 3 | \(1/2+1/2+1/3=4/3\) | 0 |
| \(b\) | \(q_1,q_2,q_3\) | 3 | \(1/2+1/2+1/3=4/3\) | 0 |
| \(c\) | \(q_3,q_4\) | 2 | \(1/3+1=4/3\) | 1: \(q_4\) |

The reachability sets are \(\{a,b\}\) for \(q_1,q_2\), \(\{a,b,c\}\) for \(q_3\), and \(\{c\}\) for \(q_4\). All three RC scores tie, although only \(c\) is immediately indispensable. Under a two-case budget, \(\{a,c\}\) or \(\{b,c\}\) covers all four problems; \(\{a,b\}\) does not.

After deleting \(a\), \(b\)'s RC becomes \(1+1+1/2=2.5\), and \(c\)'s becomes \(1/2+1=1.5\). Deleting both \(a\) and \(b\) based on their original zero removal effects would lose \(q_1,q_2\).

### Shared credit versus marginal removal

Define \(F(S)\) as the number of problems solved under the example's fixed relation. The following identities are our derivations:

\[
F(S)=\left|\bigcup_{c\in S}\operatorname{Cov}(c)\right|,
\qquad
\sum_{c\in S}RC(c)=F(S).
\]

The second equality follows because each covered query with \(m_q\) solvers contributes \(1/m_q\) to each of those \(m_q\) solvers, totaling 1. RC therefore distributes total covered-problem credit.

The marginal coverage lost on removing \(c\) is instead:

\[
\Delta_F(c\mid S)=F(S)-F(S\setminus\{c\})
=\left|\operatorname{Cov}(c)\setminus
\bigcup_{j\in S\setminus\{c\}}\operatorname{Cov}(j)\right|.
\]

It counts uniquely supplied coverage in the current set. In the example, the three removal losses sum to only 1, although \(F(S)=4\). Marginal removal effects do not generally add up to total competence.

**Connection to loss differences.** If we define a binary loss as 1 for an unsolved reference problem and 0 otherwise, then \(L_D(S)=1-F(S)/\lvert D\rvert\). Substitution gives:

\[
L_D(S\setminus\{c\})-L_D(S)=\frac{\Delta_F(c\mid S)}{\lvert D\rvert}.
\]

This explains a precise connection between coverage loss and the form of EnergyCompress's score. It uses a binary coverage loss **for explanation**; it is not EnergyCompress's published hinge loss. With real predictions, removing a harmful case can improve answers and produce a negative loss difference, which the monotone coverage model cannot express.

### Connections that do not imply equivalence

| Connection | Where the methods meet | What must remain distinct |
|---|---|---|
| \(RC\) and \(CM\) | Both relate coverage to alternative solvers | RC discounts each covered query separately; CM uses one denominator for the case's own problem. They coincide if every covered query has the same solver count as that case's own problem, but not generally. |
| \(RC\) and \(RP\) | Both aggregate contributions over covered queries | Solver counts versus adaptation costs. More alternatives do not necessarily mean equally cheap alternatives. |
| RBM and BBNR | Both inspect retrieved cases and label agreement | RBM counts disagreement even if the full answer is correct; BBNR liability requires a wrong aggregate answer and adds a preservation test. |
| Alignment and RelCBR | Both use similarity-weighted neighboring solutions | Alignment evaluates fixed neighborhood agreement. RelCBR estimates interacting reliability weights and uses them inside predictions. |
| RC and FootprintCA | Both account for alternative ways to solve problems | FootprintCA additionally represents AND requirements; an individual case's coverage list alone cannot encode these. |
| AGCBM and removal loss | Both can use prediction errors | AGCBM credits participation in source/rule uses. Removal loss compares two system states. Participation credit is not a counterfactual effect. |
| EnergyCompress and fidelity scores | Both can identify cases worth examining | Improving reference loss does not prove factual correctness. Fidelity and measured task utility can disagree. |
| NEFCS and static scores | Both use performance evidence | NEFCS makes the evidence recent and uncertain, and changes case availability reversibly. This is more than adding an age penalty. |

**Complementarity example.** If a new problem requires \(a\) and \(b\) together, neither solves it alone. Both are necessary when the pair is present. This differs from the duplicate cases above, where either suffices. A method needs the actual combination rule or a system-level evaluation to distinguish these situations.

### Questions to use while reading any formula

1. What is the reference population: original cases, held-out selection queries, or recent operational queries?
2. What establishes success: label agreement, aggregate correctness, tolerable error, an adaptation relation, or a margin?
3. Is the quantity about the case's own neighbors, queries retrieving the case, or predictions after removal?
4. Is it a sum, an average, a ratio, a fitted weight, or a difference between system states?
5. Which other cases, similarities, adaptation rules, and model parameters are assumed fixed?
6. Does selection add cases, delete cases, or change their availability? When is evidence recomputed?
7. Is it intended to measure fidelity, shared competence, unique competence, adaptation effort, or uncertainty?

These questions also help prevent double counting later. For example, task-loss removal and coverage preservation may already respond to the same loss of useful support. Whether both merit separate terms depends on the distinct requirement each would enforce.

## Research objective: recover prior methods as special cases

The PI confirmed on 2026-09-20 that this is part of the intended contribution. The aim is to derive a general case maintenance score and method whose special cases recover established approaches. The framework should explain their shared structure, make their different assumptions explicit, and support useful extensions for NN-kNN.

### First component-level reduction: relative coverage

Let \(D_+\) contain reference queries with at least one successful solver in the current case set. Use the same reference queries and solving relation for both methods. Suppose:

1. Each successful solver's stored label agrees with the query's reference label, so the relative-coverage and C/H success definitions agree.
2. Activation is uniform among those solvers and zero elsewhere:
   \[
   a_i(q)=
   \begin{cases}
   1/\lvert\operatorname{Reach}(q)\rvert,&i\in\operatorname{Reach}(q),\\
   0,&\text{otherwise}.
   \end{cases}
   \]
3. C/H uses the stored-label classification baseline. An identity or disabled adapter is sufficient for this specialization; it does not restrict the accepted use of post-adaptation contributions in the general design.

Substitution gives:

\[
C_i=\sum_{q\in D_+}a_i(q)\mathbf1[y_i=y_q]
=\sum_{\substack{q\in D_+\\i\in\operatorname{Reach}(q)}}
\frac{1}{\lvert\operatorname{Reach}(q)\rvert}
=RC(i),\qquad H_i=0.
\]

The accepted rule using the combined adapted outcome gives the same reduction when \(s(q)=1\) for every query in \(D_+\). With the same uniform activation over successful solvers, \(C_i=\sum_{q\in D_+}a_i(q)s(q)=RC(i)\) and \(H_i=0\). An identity adapter with only correct-label cases receiving activation satisfies this condition in classification.

This is an exact algebraic recovery of the **relative-coverage score by the C component** under those assumptions. No choice of the final comprehensive score has been made. To claim that the eventual case maintenance score recovers RC, we must additionally show that it reduces to this C component under the declared settings.

**Boundary conditions.** For uncovered queries, RC assigns no credit. Normalized NN-kNN activation still sums to 1 if cases are eligible, so equivalence requires restricting the comparison to \(D_+\) or explicitly allowing zero assigned credit on uncovered queries. Exact zero activation outside the solver set also requires a suitable mask or sparse allocation; finite unmasked softmax does not supply it. Identifying successful solvers using reference answers is an offline scoring construction, not a query-time assumption that an unknown answer is available.

### What counts as recovering a prior method?

| Level of claim | What must be shown |
|---|---|
| Same underlying quantities | A precise correspondence between reference problems, success evidence, neighborhoods, and case contributions. |
| Exact score recovery | Substituting the specified settings into the new score gives the published score, with the same sign, scale, and edge-case conventions. |
| Ranking equivalence | A stated transformation preserves case order. This weaker claim does not automatically preserve numerical thresholds or combined objectives. |
| Full maintenance-method recovery | The score and the add/delete procedure, update schedule, tie rules, protection rules, and stopping/selection criteria reproduce the earlier method's decisions. |
| Useful neural extension | The broader choices enabled by NN-kNN improve a stated task or maintenance objective under matched evaluation; the reduction itself does not establish that empirical benefit. |

For RC-CNN, reproducing RC scores alone does not reproduce its condensed-set construction [S4]. Likewise, matching EnergyCompress's case competence does not reproduce its method unless its sequential deletion and reference-accuracy selection are also matched [S15].

### Further reductions to investigate

| Prior approach | Candidate connection to investigate | Present status |
|---|---|---|
| Relative coverage [S4] | Uniform shared activation and matching successful-solver evidence | Exact C-component reduction shown above; final-score and selector mapping pending |
| Reputation-Based Maintenance [S11] | Signed agreement accumulated over the same reverse-neighbor exposure events | A general formulation must account for its event weighting; per-query normalized weights cannot silently replace unit event counts |
| Relative performance [S7] | Contributions measured through adaptation-cost savings over covered problems | Target for derivation; cost reference and worst-alternative denominator must match |
| AGCBM [S8] | Separate source-use and adaptation-rule evidence, with reciprocal-error credit and role weighting | Target for derivation; no reduction to a single selected NN-kNN formula established |
| EnergyCompress [S15] | Case contribution as a removal-induced difference in a specified set-level loss | Target for derivation; energy, margin, reference set, and selector must match |

These are targets for the proposed theory, not claims that a formula already encompasses them. The research can also identify methods whose assumptions require a different selector or cannot be expressed by the chosen score family.

## Implications to carry into brainstorming

These are analytical implications, not adopted changes to the architecture.

1. **Start from what the system must preserve.** Define the reference tasks, task loss or success criterion, and important regions. A global average can hide a serious loss on a rare group.
2. **Coverage with redundancy can be measured jointly.** Relative coverage shares credit among alternatives. A removal-loss score measures the net consequence in the current set. They answer different questions and can be compared rather than added automatically.
3. **Recompute after changes.** Two duplicates may each appear dispensable while both are present. Removing both based on stale scores can erase a capability. Complementary cases create the opposite problem: neither appears useful in isolation.
4. **Distinguish fidelity from use value.** A correct case can be redundant. A wrong case can appear helpful under a poor metric or a corrupted reference set. A single reported score can still have separate inspectable components and protection constraints.
5. **Condition on the neural state.** NN-kNN learns representations, feature weights, and per-case biases, then computes query-specific activations. Freeze a declared checkpoint for a removal comparison. Removing a case must recompute eligible retrieval and normalized activations, not just subtract its old weighted vote.
6. **Separate immediate and training effects.** Frozen-state removal measures case knowledge influence. Effects acquired through training concern model parameter influence and require controlled checkpoints or retraining. Case-layer removal does not undo training already performed with that case.
7. **Account for reuse (PI refinement, 2026-09-20).** For a combined adapted prediction, update all participating cases from the final outcome, in proportion to their normalized activations. Accumulate these updates over queries so maintenance evaluates retrieval and reuse together. Test whether the estimates become more useful as evidence accumulates; repeated retrieval of the same case groups can preserve mistaken assignments. Keep the adaptation correction penalty and pre/post diagnostics. The full case maintenance score remains open.
8. **Keep evidence roles separate.** A reference set used to choose memory is selection/training data. It is not an untouched final evaluation set. Specify self-retrieval exclusion, class/domain weighting, evidence counts, and temporal windows.
9. **Budget the maintenance itself.** A repeated leave-one-case-out computation may overwhelm retrieval savings. Sampling reference queries, restricting candidates, or less frequent updates are possible implementation choices to test, not literature guarantees.

The most direct existing families for later comparison are RC-style competence selection, liability/reputation/reliability methods, and EnergyCompress-style removal-loss scoring. Adaptation-aware and compositional methods identify additional capabilities a neural version may need. This survey does not select a winner or assemble a new weighted formula.

## Verification notes

- The 2026-09-19 expansion rechecked definitions against the archived texts. EnergyCompress's Table 1 and NEFCS's confidence-interval equation were also rendered and visually inspected. Worked examples were checked arithmetically.
- Common-notation definitions, the shared-credit identity, the binary-loss connection, and illustrative budget inequalities are explanatory restatements or derivations. They are distinguished from the papers' own scoring formulas and from future NN-kNN design choices.

- Equations and algorithms in the two seed PDFs were read from publisher copies. EnergyCompress's equation page was also rendered and inspected.
- Equation pages for relative performance, AGCBM, FootprintCA, RBM, and RelCBR were rendered and inspected to check signs, denominators, and constraints.
- RBM's original definition uses reverse-neighbor exposure (`c in N_k(q)`). The 2024 survey's compact presentation can obscure that direction; use the original for reproduction.
- The 2017 IJCAI recurrence and the longer AI Communications paper are different source versions. Only the identified IJCAI recurrence is reproduced here.
- No benchmark result in these papers establishes effectiveness of the proposed full-cycle neural CBR system. No implementation or empirical experiment was performed in this survey.

## Sources

- **S1.** Juarez, J. M., Craw, S., Lopez-Delgado, J. R., and Campos, M. (2018). *Maintenance of Case Bases: Current Algorithms after Fifty Years*. IJCAI, 5457-5463. [Publisher record](https://www.ijcai.org/proceedings/2018/770), [PDF](https://www.ijcai.org/proceedings/2018/0770.pdf). User-supplied seed; full text checked.
- **S2.** Smyth, B., and Keane, M. T. (1995). *Remembering To Forget: A Competence-Preserving Case Deletion Policy for Case-Based Reasoning Systems*. IJCAI. [Academic-hosted paper](https://folk.idi.ntnu.no/agnar/CBR%20papers/Smyth_1995_Remembering.pdf). Full text checked.
- **S3.** Wilson, D. C., and Leake, D. B. (2001). *Maintaining Case-Based Reasoners: Dimensions and Directions*. Computational Intelligence 17(2), 196-213. [Publisher](https://doi.org/10.1111/0824-7935.00140), [paper](https://www.cse.ust.hk/~qyang/Docs/2001/maintaincbr.pdf). Framework checked through publisher abstract and indexed paper passage; not a full formula audit.
- **S4.** Smyth, B., and McKenna, E. (1999). *Building Compact Competent Case-Bases*. ICCBR, 329-342. [Indexed original paper](https://citeseerx.ist.psu.edu/document?doi=edb4fb6f9f9ba1ec7bbbb2f91fb94db430ca89e1&repid=rep1&type=pdf). Definition and procedure verified from indexed passages; full download unavailable.
- **S5.** Delany, S. J., and Cunningham, P. (2004). *An Analysis of Case-Base Editing in a Spam Filtering System*. ECCBR. [Institutional paper](https://www.scss.tcd.ie/publications/tech-reports/reports.04/TCD-CS-2004-29.pdf). Full text checked.
- **S6.** Chebel-Morello, B., Haouchine, M. K., and Zerhouni, N. (2015). *Case-based maintenance: Structuring and incrementing the case base*. Knowledge-Based Systems 88, 165-183. [Publisher](https://doi.org/10.1016/j.knosys.2015.07.034). Abstract checked; equation verified through S1 and S13.
- **S7.** Leake, D. B., and Wilson, D. C. (2000). *Remembering Why to Remember: Performance-Guided Case-Base Maintenance*. EWCBR. [Author manuscript](https://homes.luddy.indiana.edu/leake/papers/p-00-03.pdf). Full text checked.
- **S8.** Jalali, V., and Leake, D. (2014). *Adaptation-Guided Case Base Maintenance*. AAAI. [Proceedings PDF](https://ojs.aaai.org/index.php/AAAI/article/download/8989/8848). Full text checked.
- **S9.** Mathew, D., and Chakraborti, S. (2017). *Competence Guided Model for Casebase Maintenance*. IJCAI, 4904-4908. [Proceedings PDF](https://www.ijcai.org/proceedings/2017/0691.pdf). Full text checked; abridged version.
- **S10.** Lu, N., Lu, J., Zhang, G., and Lopez de Mantaras, R. (2016). *A concept drift-tolerant case-base editing technique*. Artificial Intelligence 230, 108-133. [Institutional record](https://opus.lib.uts.edu.au/handle/10453/44129), [accepted manuscript](https://opus.lib.uts.edu.au/bitstream/10453/44129/1/AI-D-13-00140.pdf). Maintenance sections checked.
- **S11.** Nakhjiri, N., Salamo, M., and Sanchez-Marre, M. (2020). *Reputation-Based Maintenance in Case-Based Reasoning*. Knowledge-Based Systems 193, 105283. [Institutional record](https://diposit.ub.edu/dspace/handle/2445/194010), [accepted manuscript](https://upcommons.upc.edu/bitstreams/3b336461-4e79-4cbf-9593-83fe47b815e1/download). Original definition and base algorithm checked.
- **S12.** Parsodkar, A. P., Deepak P, and Chakraborti, S. (2022). *Never Judge a Case by Its (Unreliable) Neighbors: Estimating Case Reliability for CBR*. ICCBR, 256-270. [Institutional record](https://pure.qub.ac.uk/en/publications/never-judge-a-case-by-its-unreliable-neighbors-estimating-case-re/), [accepted manuscript](https://pureadmin.qub.ac.uk/ws/portalfiles/portal/356846149/ICCBR_2022_paper_60.pdf). Objective, constraints, and algorithm checked.
- **S13.** Parsodkar, A. P., Deepak P, and Chakraborti, S. (2024). *Navigating the Landscape of Case Fidelity and Competence in Case-Based Reasoning*. AI 2024, 235-249. [Institutional record](https://pure.qub.ac.uk/en/publications/navigating-the-landscape-of-case-fidelity-and-competence-in-case-/), [accepted manuscript](https://pureadmin.qub.ac.uk/ws/portalfiles/portal/611227453/SGAI_2024_8th_July_1_.pdf). Full text checked; CC BY manuscript.
- **S14.** Leake, D., and Schack, B. (2015). *Flexible Feature Deletion: Compacting Case Bases by Selectively Compressing Case Contents*. ICCBR. [Author manuscript](https://homes.luddy.indiana.edu/leake/papers/p-15-04.pdf). Scope and mechanism sections checked.
- **S15.** Badra, F., Marquer, E., Lesot, M.-J., Couceiro, M., and Leake, D. (2025). *EnergyCompress: A General Case Base Learning Strategy*. IJCAI, 4339-4346. [Publisher record](https://www.ijcai.org/proceedings/2025/483), [PDF](https://www.ijcai.org/proceedings/2025/0483.pdf). User-supplied seed; full text checked.
