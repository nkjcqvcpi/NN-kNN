# Actual lexical need matching and a ranking counterexample

`model/t3/lexical.py` supplies a fixed256-dimensional SHA256 lexical count/L2
representation,a state-versioned query module,a BM25 control and declared
candidate pools. All NN distances and training losses use the maintained core.
`tools/t3_need_matching.py` actually optimizes256 shared diagonal feature weights
with Adam; it freezes case biases,per-case weights,raw cases and representation.
It learns provided-source-passage identity,not answer or downstream utility.
No public dev annotation trains or selects the model; no global promotion occurs.

128 fitting and64 tuning SQuAD train questions use article-disjoint groups and
prospective ID samples. Each conditional pool contains its supplied passage and
63 ID-hashed negatives from18,896 train paragraphs. This is conditional candidate
selection,not corpus-wide retrieval. Three seeds×20 fixed epochs×128 updates
use .03 Adam,clamp .05..20 then mean-normalize weights to1. Choose the lowest
held-out article mean need loss,including epoch0. This selection rule was fixed
before observing outcomes; it must not be silently replaced after seeing recall.

| Condition | Tune need loss | Source recall@1 | Source recall@2 |
|---|---:|---:|---:|
| Fixed lexical NN | 4.03579 | .421875 | .546875 |
| Learned seed8,epoch17 | 4.00877 | .375000 | .484375 |
| Learned seed9,epoch14 | 4.00813 | .390625 | .468750 |
| Learned seed10,epoch19 | 4.00892 | .390625 | .484375 |
| BM25,k1=1.2,b=.75 | not the NN loss | .937500 | 1.000000 |

All learned metrics reduce held-out softmax loss but reduce top-two recall.
Their parameter deltas L2 are16.54–16.82; all7,680 training steps have nonzero
actual gradients. This is a real learned but ineffective ranking result,not an
untrained retriever accidentally called learned. Need-loss improvement does not
establish downstream improvement. Hash collisions,lexical representation,
paragraph/query mismatch and loss-versus-ranking alignment need separate tests;
this pilot does not identify one as the sole cause. BM25 is a required strong
control; do not hide its advantage or promote the learned metric on loss alone.

Three originals and three complete actual retrainings reproduce all selected/
final feature weights,Adam states,training-event records and learning curves
exactly. Independent manual squared feature contributions,weighted distances and
full-bank softmax loss match on all7,680 training and12,672 evaluation calls.
Manual gradients agree within rtol1e-5/atol1e-7,max absolute5.96046e-8. The first
zero-tolerance gradient check failed at float32 reduction-rounding2.98e-8; its
failed attempt is preserved. Gradient tolerance does not alter actual training;
entire learned/optimizer state equality remains exact. Two focused tests verify
source-based pools,BM25,real core gradients,frozen other parameters and audited
retrieval geometry;35 T3 tests passed at the initial package boundary.

Frozen source `2a947b1490148acb0643c08e2197a0800407e0ff62c9d1bd90e6d8b2be442a05`,
138 files,ZIP SHA256
`fcf5a39cdeb10b66737e3539726b97870d29b3243b999a8afe98e9936b9904a0`.
Source and fullZIP contents validate. Task evidence:
`work/t3-need-matching-20261005/`,
`work/t3-need-matching-replayed-20261005-gradient-tolerance/`,
`outputs/neural-cbr-t3-need-matching-verification.json`.

Next public-host controls retain the prespecified seed8 snapshot and report this
failure rather than choosing a dev-successful seed. Prospective alternatives may
align training/checkpoint selection with constrained retrieval budgets,improve
representations and collect objective counterfactual utility. They must remain
separate experiments; typed heads and full feedback lifecycle remain open.
