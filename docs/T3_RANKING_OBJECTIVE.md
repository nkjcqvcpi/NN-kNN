# Prospective ranking objective and checkpoint selection controls

The previous loss-selected lexical model lowered softmax loss while degrading
top-two source recall. Two prospective controls keep its128 fitting/64
article-disjoint tuning questions,64-case pools,256 dimensions,three seeds,
20 epochs,Adam .03 and projection unchanged:

1. Keep softmax training,but select maximum tune recall@2,then recall@1,then
   minimum loss,including epoch0.
2. Use a smooth ranking loss against the eight closest actual-core negatives,
   margin .2,and the same recall-based selection. Positive source ID comes from
   the supplied passage; no answer string or public dev annotation trains it.

`need_loss` explicitly exposes these exploratory objectives. Full-bank geometry
is required; hard negatives use actual core distances,not a separate encoder.
This is need matching,not objective downstream usefulness or an approved global
policy. Defaults remain the original softmax/tune-loss configuration.

| Control | Selected epochs,seeds8/9/10 | Tune recall@1 | Tune recall@2 |
|---|---|---|---|
| Original loss-selected softmax | 17/14/19 | .375/.390625/.390625 | .484375/.46875/.484375 |
| Softmax,recall selection | 1/0/0 | .421875/.421875/.421875 | .546875/.546875/.546875 |
| Hard negatives,recall selection | 3/4/1 | .46875/.484375/.515625 | .65625/.671875/.65625 |
| BM25 control | no training | .9375 | 1.0 |

Selection alone prevents the observed recall downgrade but does not improve
recall: seeds9/10 select the original untrained epoch0. Hard negatives improve
all three tune recall@2 values over the fixed .546875 baseline but remain weaker
than BM25. Their selected parameter L2 changes are9.50/10.48/5.03. This is an
exploratory tuning result; do not call it public-dev performance or a downstream
utility improvement. Loss values from different objectives are not comparable.

Six original trainings and six complete actual retrainings verify15,360 nonzero
gradient updates and25,344 evaluation calls. Manual actual-core feature
contributions,distances and both losses match exactly; independent float32
gradients differ by at most5.96046e-8 (rtol1e-5/atol1e-7). Every selected/final
weight tensor,Adam state,learning curve and training record matches exactly.
All three softmax final weights/Adam states match the earlier loss-selection
package: changing selection does not change training.213 joint tests pass at
this boundary,including the direct hard-negative formula test.

Source `e3c33e81d687fb6ebc2c16d00f6a3b678d8dbd67875afb269a5e6e0445b9f0c7`,
139 files; ZIP SHA256
`237b95cdc881473646af51f251b1fcac0b6f5d0325ced5587b088ec408a3aa1b`.
Full source/ZIP bytes validate. Task artifacts:
`work/t3-hard-negative-20261005/`,`work/t3-softmax-recall-20261005/`,
their `-replayed/` counterparts and
`outputs/neural-cbr-t3-ranking-objective-verification.json`.

Proceed with independent public-host comparisons,resource accounting and
objective exact-set utility. Broader representations,typed heads,calibration,
poisoning/rollback and internal/combined integration remain open. The original
seed8 snapshot stays in the first public control so this later tuning result
cannot retrospectively improve that condition.
