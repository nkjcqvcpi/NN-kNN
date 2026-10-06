# Public metric selection provenance

The public host driver previously described every supplied checkpoint as seed8,
tune-loss-selected. That literal becomes false for actual-need or hard-negative
checkpoints. Evaluation now reads the training protocol, aggregate run summary
and seed summary, requires their exact agreement and matches the checkpoint
SHA256 to exactly one recorded run. It records the actual seed, selected epoch,
objective, selection rule, query source/binding and training source/data hashes.
Epoch0 remains explicitly recorded; it is not described as a trained improvement.
The retriever version also uses the actual recorded seed.

The public driver rejects training-data or fit/tune split differences from the
experiment manifest and requires recorded absence of public dev annotation
training. An actual-need claim requires its query binding. Replaced checkpoint
bytes, edited selection records and mismatched query claims are rejected.
These bindings verify recorded provenance; they do not prove that the caller
selected a seed before viewing public results. The historical literal tune-loss
rule is preserved and normalized explicitly rather than silently relabeled.

All57 T3 tests pass. Six real historical checkpoints independently pass these
bindings and exact data/split checks: loss-selected seeds8/9/10 epochs17/14/19
and hard-negative recall-selected seeds8/9/10 epochs3/4/1. Task evidence:
`outputs/neural-cbr-t3-metric-provenance-verification.json` and
`work/check_t3_metric_provenance.py`. The first test attempt failed because the
default Windows pytest temporary directory denied access; rerunning in a new
task-local temporary directory passed. No shared temporary files were removed.

Already-running request/training experiments retain their frozen147-file source.
This change prepares subsequent public comparisons; it is not a completed new
host evaluation, metric promotion or proof of outcome benefit.
