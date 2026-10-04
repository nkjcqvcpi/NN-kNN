# Adapted removal with fixed continuation budgets

The September 20 PI note permits a higher-cost retraining comparison and asks
for final adapted prediction loss in removal influence. The efficient frozen
audit remains a separate path. `model/t1/adapted_retraining.py` implements
independent model, external adapter and selected Adam forks. Original training
queries retain stable IDs and leave-one-out exclusion after case compaction.
Nominal vocabularies and original case bindings remain fitted on training data.

Retrieval-only continuation minimizes final classification loss through a frozen
adapter; adapter-only continuation uses the declared v0 combined/difference/class
objective through frozen retrieval. With both budgets positive, single epochs
alternate starting with retrieval. The fixed final epoch is saved; validation
and test streams are not accessed during continuation or candidate selection.
This version supports classification with external adapters. Regression and
shared/internal adapter variants remain separate extensions.

Every round refreshes eligible deletion candidates and a full-memory training
control from the same accepted state, moments, RNG and minibatches. Candidate
loss uses that candidate's continued adapter and the complete declared reference
denominator. Selection excludes adaptation penalties, preserves protected IDs
and cohort floors, uses deterministic stable-ID ties, and accepts only within
the cumulative final-loss increase relative to the original adapted checkpoint.
Rejected forks do not change accepted state. Archive records retain the removed
case's state before that round's continuation.

The runner also trains unchanged full memory from the original checkpoint for
the same number of accepted phase blocks. Equal examples, phase budgets and
Adam updates do not imply equal retrieval cost when capacity differs. The trace
reports all candidate/control/accepted updates, query-case pairs and elapsed
time. Training histories expose the final classification loss and the adapter
objective separately; no calibrated minimal-correction penalty is introduced in
this v0 continuation comparison.

A protected random-deletion comparator reaches the adaptive method's actual
final capacity, with the same number of training phase blocks and examples.
Its deletion sequence uses a separate seed-local generator and preserves the
same protected IDs and cohort floors. It does not use the final-loss budget to
choose or stop; the runner explicitly records whether its resulting loss violates
that budget. Its checkpoint and metrics are saved independently. This comparator
matches accepted training and final capacity, while adaptive candidate search
still costs more computation.

```powershell
.venv\Scripts\python.exe tools/t1_run.py configs/t1/alignment_adapted_retraining.yaml --out <isolated-output-directory>
```

The initial configuration is an exploratory 18-training-case Iris pilot, three
seeds and three scopes, target capacity 12, one epoch per enabled component per
trial, learning-rate scale .1, combined adapter loss, minimum one case per class,
and cumulative allowed loss increase .03. These are declared pilot choices,
not PI-approved confirmatory thresholds. Checkpoints save initial, final and
matched-full model/adapter/optimizer states, source ZIP, raw splits, actual test
retrieval events, scoring traces and budgets.

Thirteen focused tests exercise actual updates in both output modes and all scopes,
source/RNG isolation and deterministic replay, frozen nominal schema after
compaction, invalid budgets/IDs, fresh first-candidate loss replay and held-out
selection rejection, and no mutation after a trained candidate exceeds budget.
protected random-control repeatability and protection rejection.
All 132 current T1 tests pass. One development seed reached
12 cases in each scope and .90 final test accuracy. The frozen nine-run pilot
now passes all independent audits: 36 stage states, 837 fresh candidate trials
and 54 full-memory round controls. A separate parent rerun of complete searches
exactly matched every final model/adapter/Adam tensor and budget in all nine runs.
The source fingerprint is
`7c937fc5eac495ecbbf6a05e3ead80c734e1c0fa76e436ac9ebad7f0ebe99972`,
103 ZIP members, SHA256
`f24cba5cef4423d2289446ec7fd752da19bad7ffda91729e2835230d1a921188`.
Earlier independent harness failures are preserved in the task outputs.

All three scopes reached 12 cases. Their three-seed mean final accuracy is .9111,
versus initial .9222 and matched random .8889. Matched full memory is .9111 for
retrieval-only/alternating and .9222 for adapter-only. Each random run violates
the declared .03 training-reference loss budget, despite some better test
outcomes. The seed-wise adaptive changes include both gains and losses. Neither
the accepted training budget nor these three seed means guarantees better test
performance. Alternating uses both role budgets and cannot be ranked as equal
compute with one-role scopes. Candidate-search computation is much larger than
accepted-model training; both are recorded separately. No general compression
benefit is claimed. Wider budgets, frozen controls, regression,
constraint/protection stress and broader datasets remain required follow-ups.
