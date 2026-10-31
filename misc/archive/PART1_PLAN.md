# Part 1 implementation plan

## Objective and scope

Reproduce the qualitative progression requested in the [take-home README](vjepa-physics-takehome-4E00/README.md): layer-wise probing, iterative nullspace probing, and multi-probe steering for direction, speed, and acceleration. Deliver reproducible figures and a concise methods/results narrative suitable for the final 15-minute presentation. Part 2 is out of scope for this first pass.

The [physics paper](https://arxiv.org/html/2602.07050v1), especially Sections 3, 5, 7 and Appendices C.11–C.12, motivates mean-pooled linear probes, orthogonal probe sequences, and subspace steering with an independent readout. We will distinguish our dataset, split, solver, and evaluation choices from the paper's choices. Its expected trends are hypotheses, not acceptance criteria.

## 1. Establish data and model contracts

- Source data: `vjepa-physics-takehome-4E00/data/`, never modified. Confirmed 1,500 direction clips, 1,536 speed clips, and 1,536 acceleration clips.
- Audit all manifests, video decoding, dimensions, frame counts, missing files, and duplicate video hashes before splitting. Summarize label frequencies, motion regimes, starting positions, and joint label distributions. In particular, direction clips mix motion regimes; scalar datasets also vary direction. Check whether scalar labels correlate with angle or position.
- Use the pinned `facebook/vjepa2-vitl-fpc64-256` snapshot recorded in `checkpoint_manifest.json`. Configuration specifies 24 transformer blocks, 1,024 hidden dimensions, 16-pixel patches, and two-frame tubelets.
- Store checkpoint, environment, caches, and activations under `/scratch/users/hmolin/jepa-interp/`; keep small source/configuration/results files in this workspace. Record dependency versions, seeds, model revision, and input hashes. Scratch artifacts should be reproducible.
- Load the encoder frozen, in evaluation mode. Use deterministic preprocessing with documented color order, normalization, tensor layout, and no random crops or flips.
- Resolve the 16-frame dataset versus 64-frame checkpoint configuration with an explicit smoke test. Preferred protocol: all 16 native frames, preserving the original time scale, if the implementation accepts variable temporal length. Verify positional encoding and processor behavior rather than silently repeating/interpolating frames. If adaptation is required, document and validate it before extraction.
- Validate hooks on each block's residual output; explicitly distinguish those outputs from any final normalization. Expect 2,048 space-time tokens for 16 frames at 256 resolution, subject to implementation verification.

Acceptance: a small GPU batch loads correctly, produces finite deterministic features at all 24 layers, and agrees with the normal encoder forward path. Run GPU work through Slurm, not on a login node.

## 2. Freeze a leakage-resistant evaluation protocol

Proposed split per dataset: A = 50% steering/probe training, B = 15% validation, C = 15% independent evaluator training, D = 20% final test. Use a fixed seed, stratify by primary label where possible, keep duplicates in the same partition, and save clip IDs. Check joint nuisance-variable balance; never assume scene IDs align across datasets.

- Layer-wise and nullspace probes fit A; regularization, layer selection, stopping, and steering settings use B.
- Independent steering evaluators fit C only. Fix their architecture and regularization through cross-validation within C; C is never used to construct steering subspaces.
- D is untouched until the protocol is frozen. Both ordinary probe scores and steering scores are reported there; no evaluator is fitted on D.
- Fit feature centering and label transformations only on the relevant fitting partition. Prefer centering without per-coordinate feature scaling for the primary experiment so the Euclidean projection geometry remains explicit.
- Primary split measures new-clip generalization over observed labels. Add a separately specified nuisance holdout or held-out label-level interpolation experiment if the audit shows adequate coverage. Do not describe a random split as out-of-distribution generalization.

## 3. Extract and cache pooled features once

Mean-pool space-time tokens at every block. Write resumable shards with clip IDs, layer convention, dtype, preprocessing configuration, and model revision. Store pooled float32 features: approximately 450 MB for all 4,572 clips × 24 layers × 1,024 dimensions. Avoid caching every token at every layer; that would be roughly 920 GB. Recompute token features only for targeted follow-ups.

Benchmark a small batch to choose memory-safe batch size and estimate full extraction time. Compare reduced-precision inference to float32 on a small sample if using it. Check shard completeness and finite values before probe fitting.

## 4. Layer-wise probing

- Fit regularized linear regression for speed and acceleration; use two outputs `(sin(theta), cos(theta))` for direction. Tune a small, fixed ridge grid on B. This solver is our efficient first-pass choice, and should be documented as a methodological variation.
- Report R² and MAE in physical units for scalar targets. For direction, report circular MAE in degrees and sine/cosine R²; define treatment of near-zero predicted vector norm.
- Compare with training-mean/constant predictions and label-shuffled probes. Inspect train–validation gaps because feature dimension is large relative to sample count.
- Plot all 24 layers, with clip-bootstrap confidence intervals on final scores. Select the intervention layer using B, preferably a common layer with strong performance across variables; use task-specific layers if the common choice is weak and disclose the selection rule.
- Add first-frame-only/repeated-frame and temporal-order controls on a small selected-layer experiment to test whether appearance or displacement shortcuts explain apparent physics decoding. These controls are separate derived inputs, not edits to the dataset.

Acceptance: three interpretable depth curves, baseline comparisons, a documented layer choice, and explicit discussion of any failure to match the expected qualitative ordering.

## 5. Iterative nullspace probing

For each target at its selected layer, repeatedly fit a fresh linear probe on projected A features, evaluate on B, and remove the readout row space. Maintain a cumulative orthonormal basis with rank-revealing SVD/QR, checking orthogonality and projection residuals. Apply exactly the same projection to all partitions.

Predefine a maximum rank (initially 128) and a validation stopping rule based on sustained near-baseline performance. Plot both probe iteration and actual removed rank: a circular probe can contribute up to two dimensions, whereas a scalar probe contributes at most one. Include rank-matched random removals and shuffled-label controls. Describe remaining decodability as evidence about linear redundancy under this protocol, not an intrinsic dimension estimate or proof that information is absent.

Save probe biases, effective weights in the original centered coordinates, cumulative bases, and all validation decisions. Mapping projected probe weights back to original coordinates is essential for the steering constraints.

## 6. Multi-probe steering with independent evaluation

Construct a basis V from successive effective probe weights. For source x and target y*, solve the stacked linear constraints for coordinates c* within V, including probe intercepts. Use a stable least-squares solve, with any regularization fixed on B. Reconstruct x* = x + V(c* - Vᵀx), preserving the orthogonal component. Unit-check the preserved component and constraint residuals.

- Sweep the number of probes (1, 2, 4, 8, 16, 32, then available ranks). For direction, include the paper's 90-degree target plus a predefined full-circle grid; for scalars, use predefined low/middle/high targets inside the observed range.
- Apply interventions only to held-out D clips for final reporting. Score target error using the independent C-trained evaluator, first reporting its ordinary predictive accuracy on D.
- Compare no intervention, one-probe steering, multi-probe steering, and norm-matched random perturbations. Include the original-label error, perturbation norm, and source-to-target breakdown so a mean improvement cannot hide failures.
- Where labels vary sufficiently, measure collateral changes in independently decoded nuisance variables. Do not report preservation metrics for constant labels.
- Present this as a pooled-representation intervention. It does not demonstrate changed videos, physical trajectories, or model rollout behavior. A later token-level intervention followed by remaining encoder blocks would be a separate, stronger experiment.

Acceptance: independent target-error-versus-rank curves for all three properties, direction wraparound coverage, scalar boundary checks, and meaningful control comparisons. A negative result remains a valid outcome.

## Execution order and deliverables

1. Verify checkpoint and GPU/environment smoke test; complete data audit and save split manifests.
2. Extract pooled activations, benchmark throughput, and fit layer-wise probes.
3. Freeze intervention layers and nullspace stopping rules from validation results.
4. Construct steering subspaces, train independent evaluators, and freeze the final evaluation configuration.
5. Run final tests once; create figures, a results table, and limitations notes.

Proposed source modules: `audit_data.py`, `extract_features.py`, `probe_layers.py`, `nullspace.py`, `steer.py`, and `make_figures.py`, backed by one experiment configuration and saved split manifests. Meaningful checks cover split disjointness, circular wraparound, projection/rank correctness, and steering algebra. Finish with 5–7 Part 1 slides: setup, depth curves, redundancy curves, steering method, held-out results, controls, and limitations. Keep outputs compatible with later spline comparisons using the same splits and evaluators.
