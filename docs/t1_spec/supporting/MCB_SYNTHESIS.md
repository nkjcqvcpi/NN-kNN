# Momentum Case Base synthesis

## Source and handling status

- Source: `resources/research/momentum-case-base/momentum-case-base-stability-paper.pdf`, supplied by the PI on 2026-09-04.
- The document identifies itself as an anonymized manuscript for review and prohibits distribution, citation, or public sharing. Keep it proposal-local and confidential. Do not use it as a public citation unless the PI later supplies an accepted or otherwise public version.
- The PI confirmed that the manuscript was submitted to AAAI 2027 and is currently under review as of 2026-09-04. It is not accepted, peer-reviewed publication evidence, or publicly citable in its current form.
- The PI states that the work is a collaboration with people from Indiana University Bloomington (IUB). Collaborator names, affiliations beyond IUB, author order, and contribution roles are not supplied.
- The PI can provide code, but no code, checkpoints, supplementary files, or independent reproduction record have been supplied yet.
- The PI confirmed on 2026-09-04 that this MCB manuscript is the same pipeline EMA-training-policy paper previously described as an improvement to NN-kNN itself.

## Mechanism established by the manuscript

MCB addresses a moving-reference problem in neural case-based models. If one changing encoder represents both queries and persistent references, training can move the reference geometry and cause stored cases or prototypes to change their relationship to a query. Some movement enables adaptation; excessive movement can destabilize prediction, explanation, and later learning signals.

MCB separates two representation paths:

- an online encoder, optimized rapidly by backpropagation and used to represent queries; and
- a memory encoder, initialized from the online encoder, excluded from gradient updates, and updated once per training iteration as an exponential moving average of the online parameters.

The manuscript instantiates the mechanism in two model families:

- **MCB-R:** The online encoder represents the query and the momentum encoder represents stored raw cases. A differentiable NN-kNN-style retrieval rule uses normalized embeddings, a learned weighted distance, temperature-scaled activations, and learned nonnegative case weights. The same-source case is masked during training; top-k cases provide the inference-time vote and local explanation.
- **MCB-P:** The online encoder remains responsible for the ProtoPNet loss and inference. The momentum encoder is used only to encode candidate patches during prototype projection. EMA updates the encoder used to construct a projected prototype, not the prototype directly.

The manuscript evaluates four aspects of the learned reference space: stored-case geometry, retrieved-neighborhood stability, prototype stability across projection events, and class geometry.

## Reported evidence

All results below are manuscript-reported and have not been independently reproduced for this workspace.

- Dataset and scope: CUB-200-2011 fine-grained bird image classification only.
- MCB-R reportedly achieves 88.9% top-1 accuracy with its ResNet-50/iNaturalist configuration and 85.8% with DenseNet-161.
- Against a matched shared-encoder baseline, MCB-R reportedly reduces adjacent-epoch representation change and retrieved-neighborhood churn and ends with 6.2 times lower NC1 and 1.3 times lower CDNV.
- Ten independently resampled case bases reportedly retain 88.9% mean accuracy with 0.03-point standard deviation, 0.999 prediction agreement, and 0.61 mean attribute overlap among top-five retrieved cases.
- Replacing the fixed MCB-R retrieval vote after training reportedly yields 88.8% with a nearest-class-mean classifier and 88.1% with a fitted linear classifier, compared with 88.9% for MCB-R retrieval.
- MCB-P reportedly reduces later projection-to-projection prototype drift and improves class geometry under L2 and cosine variants. In the L2 setting it improves mean top-1 accuracy from 68.67% to 74.33%, a 5.66-point gain; in the cosine setting it changes mean accuracy from 85.81% to 86.07%.

These results support the narrower claim that the proposed EMA memory encoder can stabilize learned reference geometry across two neural case-based image-classification families without freezing representation learning. They do not establish effectiveness in sequential decision-making or language/agent systems.

## CAREER relevance

MCB is potentially useful in three ways:

1. **Preliminary foundation:** It gives the PI a concrete mechanism, diagnostics, and reported evidence for improving NN-kNN itself, strengthening the progression from classification and regression toward more dynamic systems.
2. **Unifying technical theme:** Its central tradeoff—preserving inspectable reference structure while allowing adaptation—aligns with the planned inspectable, editable, quality-aware, and adaptive case-memory layer.
3. **Research opportunity:** Applying, modifying, or stress-testing this two-timescale memory idea in RL, retrieval/RAG, and agents could become proposed CAREER work if the scientific questions and novelty relative to the manuscript are made explicit.

The PI intends MCB to become a feature of the NN-kNN core and to be evaluated as the framework expands, including within the NN-kNN-RL line. The PI also judges that the basic first step is mostly complete in the submitted paper. Consequently, MCB is preliminary evidence and one candidate mechanism within T1, not the whole proposed thrust. The CAREER contribution must go beyond repeating the paper by testing generality across prior NN-kNN settings and developing stable, adaptive, editable, and quality-aware case memory under harder conditions. The expectation that MCB will improve NN-kNN over the long run remains a research hypothesis requiring controlled evaluation.

An immediate, pre-award benchmark pilot will compare MCB with the original shared-encoder NN-kNN core on existing classification benchmarks. Regression is excluded from the first pilot and may be considered after the shared-core classification integration is stable. This can strengthen preliminary evidence and reveal where MCB helps, has no effect, or slows useful adaptation. The pilot plan is maintained in `MCB_BENCHMARK_PILOT.md`.

## Boundaries and unresolved items

- Describe the manuscript only as submitted to AAAI 2027 and under review as of 2026-09-04. Do not call it accepted, peer reviewed, published, or publicly citable unless its status changes.
- Do not infer the anonymous author list or individual contributions.
- Do not claim that MCB addresses poisoned cases, bad knowledge, domain shift, human edits, healthcare, RL, LLMs, or agents; none is tested in the supplied manuscript.
- Do not equate MCB encoder EMA with mutable critic-label EMA or lagged target-critic EMA in the current RL implementation.
- `[NEEDS INPUT: update the manuscript's status when an AAAI 2027 decision is received]`
- `[NEEDS INPUT: collaborator names, affiliations, and contribution roles when the PI is ready to provide them]`
- `[NEEDS INPUT: code, configuration, checkpoints, and reproduction evidence before using implementation details as verified feasibility]`
- `[NEEDS INPUT: define the first MCB-to-RL integration point, baseline, and stability/performance success criteria when the RL experimental plan is selected]`
