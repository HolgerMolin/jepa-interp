# Audit of the unexpectedly early direction decoding

These are post-hoc diagnostics on the previously viewed dataset, not a new confirmatory evaluation. The original features, splits, and primary probe results were preserved. Nullspace probing and steering remain unrun.

## Implementation checks

- Independent scikit-learn SVD ridge fits at blocks 1 and 19 reproduced all three targets’ saved predictions. Maximum absolute difference: 8.43e-11.
- Fresh inference using the separate hidden-state API agreed with cached raw features at blocks 1, 8, 19, and 23 for three clips in each dataset. Maximum absolute feature difference: 0. This checks layer indexing and cache association independently of the original hooks.
- Pixel centroid displacement agreed with metadata direction for 35 sampled clips: median angular error 0.466°, maximum 3.724°. One additional sampled clip lacked a visible disk at its endpoint. This is a sampled direction-label check, not an audit of all scalar physical units.
- The disk decodes orange in RGB, contrary to the dataset reference’s description of blue. The original RGB loader was correct; no color swap was applied.
- Primary fitting used A only, hyperparameter/layer selection used B, and test metrics used D. C remains reserved. File hashes and IDs were disjoint across partitions. These checks do not exclude every near-duplicate or rendering confound.

## Properly refitted direction baselines

Unlike the initial sensitivity controls, these models were each fitted on A and tuned on B using their own input representation. They are evaluated on D. Normalized features apply the checkpoint encoder’s LayerNorm to each token before mean pooling; patch embeddings precede all transformer blocks. Static inputs repeat only the first frame 16 times.

| Representation | Test R² | Circular MAE |
|---|---:|---:|
| Original raw block 1 | 0.838 | 13.53° |
| Original raw block 19 | 0.989 | 3.22° |
| patch_embedding_only | -0.093 | 69.87° |
| static_block1 | -0.014 | 89.67° |
| static_block19 | -0.011 | 86.19° |
| normalized_block1 | 0.842 | 12.91° |
| normalized_block8 | 0.964 | 5.38° |
| normalized_block19 | 0.988 | 3.35° |

## Repeated label permutations

The original baseline used one shuffle and chose its regularization against true validation labels, making it an imperfect null reference. This diagnostic fixes regularization at the primary setting and averages 30 independent training-label permutations. The primary real-label fits are unaffected.

- Block 1: mean MAE 90.01°, mean R² -0.525.
- Block 19: mean MAE 90.43°, mean R² -0.368.

## Motion-regime transfer

Fit on A clips from one regime, choose regularization using B from that same regime, and evaluate D from the other regime.

| Transfer | Block | Test clips | R² | Circular MAE |
|---|---:|---:|---:|---:|
| velocity->acceleration | 1 | 148 | 0.684 | 22.83° |
| velocity->acceleration | 19 | 148 | 0.960 | 5.51° |
| acceleration->velocity | 1 | 144 | 0.619 | 12.31° |
| acceleration->velocity | 19 | 144 | 0.966 | 5.10° |

## What remains uncertain

The audit supports the computational correctness of this first-pass probe study, but does not establish a tightly matched paper reproduction. [Appendix B of the paper](https://arxiv.org/html/2602.07050v1#A2) specifies grouped five-fold cross-validation and a learning-rate/weight-decay sweep; our initial experiment used one label-stratified split and closed-form ridge. The paper does not sufficiently specify the grouping variable in that paragraph to reconstruct it without further evidence. The take-home data also use a simpler renderer and different sampling.

[Meta’s official intermediate-output API](https://github.com/facebookresearch/vjepa2/blob/main/src/models/vision_transformer.py) applies LayerNorm before returning block outputs. Our initial raw-residual hooks are internally consistent, but should not have been treated as automatically equivalent to that API. The normalized comparison above tests this distinction; it does not establish which extraction convention the paper used.

We preserved native frames rather than the checkpoint processor’s resize/center-crop default. That spatial-preprocessing choice has not been isolated in this audit. Nor have we completed grouped cross-validation, random-initialization controls, per-patch spatial generalization, or a full faithful optimizer replication. Do not tune toward an expected curve or present the initial result as a settled contradiction of the paper.
