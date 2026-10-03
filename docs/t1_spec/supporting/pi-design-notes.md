# PI case-maintenance design notes

Source: PI messages in this workspace conversation, 2026-09-20. Original wording, numbering, and spelling are preserved below. The two subsequent clarification replies are recorded separately. This is a source record, not an edited design specification.

## Original message

Here are some important thoughts I have since reading your survey. Try to understand every details of my description. If anything vague, ask me. I propose several measures to try below.

4. One possible formula, is to do s4, the coverage-to-reachability ratio, we can do so by defining "solve()" as anything that correctly contributes, beyond an activation threshold, to a prediction of a case, and then we can define "reach()" and "cover()" accordingly, then we can calculate the ratio as score for all cases. This can be a valid measure.

5. One possible formula:

using s12: we are already doing this by learning parameters in our model. , even better we learn everything in backprop, principled way!

High cost but ultimate method:
Measure influence of removing a case.  We may call this case influence score. (what do you think? any better name?)
The algorithm would be similar to algorithm in S12:
Measure current loss,
tentatively remove a case,
Measure new loss,
The difference is the influence.
We will choose the best influence to remove cases. This may be no damage to performance, or minimal damage to performance with good case base reduction.
This is similar to your description here: "For NN-kNN, a corresponding removal comparison would keep the chosen model checkpoint fixed, remove the case, recompute retrieval and normalized activations, and run reuse again. When adaptation is enabled, evaluate the final adapted outputs in both conditions."
note, if possible, retraining can be enabled, that could potentially condense the case base further, but surely it is going to cost more computations for retraining.
Also note, this can be very inefficient if the influence is calculated by running the whole evaluation/training set(with leave one out) again. Instead, during forward pass, we can use record queries that a case activates beyond a threshold, and then when calculating the influence of a case, we just need to rerun test on these queries using the case base without the query. We don't need to retrain model for efficiency. And we allow model to retain parameters trained using the case because such knowledge can be integrated even though the case is now removed.

2.In our current formula of maintenance score, Qi and Bi are kind of overlapping. A good Qi likely correlates with a good Bi. So doing Qi* Bi with some weights in their exponent is probably not wise. But I do like the idea of Ci and Hi, or Qi, seperate from Bi. We can see a lot of papers are doing similar things, and ours is more smooth and fine-grained weighted by activation than just counting good/bad votes.

I think we could just use a variant of Qi or just Qi. as a measure

Then we could just use Bi as an alternative measure, but not Qi and Bi at the same time.

3. The current design, no matter using Qi or Bi, is more about the competence/usefulness of the cases in the full nn-knn cycle. It is also related to coverage. A good coverage case will be retrieved often and trained well, with high bias. However, our current design does not account for necessity/redundancy.
   Similar to the idea in point 1, during forward pass, we can record the queries the case activates for, including their activation. This activation map is therefore a signature of how useful this case this. We can then find redundant case by comparing their activation maps, if they activate in similar scales for similar cases, then they are not needed.
   However, this may not be accurate measure of redundancy because during training, if two cases are duplicate of each other, the model might just learn to ignore one and use the other, so their activation map will be drastically different.
   Let's put this off on the side.

4.

With that said, I think all the papers you surveyed more or less map to what we discussed above. They either can be a specific variant of one of the ones above, or they can be replicated in our full neural CBR.

## Clarification 1: solve relation

Assistant question: For the coverage/reachability ratio, should case i count as solving query q when its activation exceeds the threshold and the final adapted prediction is correct—even if i’s stored label disagrees—or must i’s own adapted solution be correct?

PI reply: Use the final adapted query outcome, consistent with our current C/H rule.

## Clarification 2: removal loss

Assistant question: For the case-removal comparison, should “loss” mean only the final prediction loss, or should it also include the adaptation penalty for large corrections?

PI reply: Final prediction loss only; report the adaptation penalty separately.
