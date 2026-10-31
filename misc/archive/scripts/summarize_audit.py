import json, hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/diagnostics'
r=json.loads((OUT/'audit_results.json').read_text()); primary=json.loads((ROOT/'results/probe_results.json').read_text())
maxsolver=max(v['max_prediction_difference'] for t in r['independent_solver'].values() for v in t.values())
maxhook=max(v['max_error'] for v in r['fresh_hidden_state_checks'])
pixel=[q['error'] for q in r['pixel_label_check'] if 'error' in q]
lines=['# Audit of the unexpectedly early direction decoding','',
'These are post-hoc diagnostics on the previously viewed dataset, not a new confirmatory evaluation. The original features, splits, and primary probe results were preserved. Nullspace probing and steering remain unrun.','',
'## Implementation checks','',
f'- Independent scikit-learn SVD ridge fits at blocks 1 and 19 reproduced all three targets’ saved predictions. Maximum absolute difference: {maxsolver:.3g}.',
f'- Fresh inference using the separate hidden-state API agreed with cached raw features at blocks 1, 8, 19, and 23 for three clips in each dataset. Maximum absolute feature difference: {maxhook:.3g}. This checks layer indexing and cache association independently of the original hooks.',
f'- Pixel centroid displacement agreed with metadata direction for {len(pixel)} sampled clips: median angular error {np.median(pixel):.3f}°, maximum {max(pixel):.3f}°. One additional sampled clip lacked a visible disk at its endpoint. This is a sampled direction-label check, not an audit of all scalar physical units.',
'- The disk decodes orange in RGB, contrary to the dataset reference’s description of blue. The original RGB loader was correct; no color swap was applied.',
'- Primary fitting used A only, hyperparameter/layer selection used B, and test metrics used D. C remains reserved. File hashes and IDs were disjoint across partitions. These checks do not exclude every near-duplicate or rendering confound.','',
'## Properly refitted direction baselines','',
'Unlike the initial sensitivity controls, these models were each fitted on A and tuned on B using their own input representation. They are evaluated on D. Normalized features apply the checkpoint encoder’s LayerNorm to each token before mean pooling; patch embeddings precede all transformer blocks. Static inputs repeat only the first frame 16 times.','',
'| Representation | Test R² | Circular MAE |','|---|---:|---:|']
for layer in [1,19]:
 q=primary['direction']['layers'][layer-1]['test'];lines.append(f"| Original raw block {layer} | {q['r2']:.3f} | {q['mae']:.2f}° |")
for name,q in r['refitted_controls'].items():lines.append(f"| {name} | {q['r2']:.3f} | {q['mae_degrees']:.2f}° |")
lines+=['','## Repeated label permutations','',
'The original baseline used one shuffle and chose its regularization against true validation labels, making it an imperfect null reference. This diagnostic fixes regularization at the primary setting and averages 30 independent training-label permutations. The primary real-label fits are unaffected.','']
for layer,q in r['permutation_null'].items():lines.append(f"- Block {layer}: mean MAE {q['mean_mae']:.2f}°, mean R² {q['mean_r2']:.3f}.")
lines+=['','## Motion-regime transfer','', 'Fit on A clips from one regime, choose regularization using B from that same regime, and evaluate D from the other regime.','', '| Transfer | Block | Test clips | R² | Circular MAE |','|---|---:|---:|---:|---:|']
for name,values in r['direction_regime_transfer'].items():
 for layer,q in values.items():lines.append(f"| {name} | {layer} | {q['test_n']} | {q['r2']:.3f} | {q['mae_degrees']:.2f}° |")
lines+=['','## What remains uncertain','',
'The audit supports the computational correctness of this first-pass probe study, but does not establish a tightly matched paper reproduction. [Appendix B of the paper](https://arxiv.org/html/2602.07050v1#A2) specifies grouped five-fold cross-validation and a learning-rate/weight-decay sweep; our initial experiment used one label-stratified split and closed-form ridge. The paper does not sufficiently specify the grouping variable in that paragraph to reconstruct it without further evidence. The take-home data also use a simpler renderer and different sampling.', '',
'[Meta’s official intermediate-output API](https://github.com/facebookresearch/vjepa2/blob/main/src/models/vision_transformer.py) applies LayerNorm before returning block outputs. Our initial raw-residual hooks are internally consistent, but should not have been treated as automatically equivalent to that API. The normalized comparison above tests this distinction; it does not establish which extraction convention the paper used.', '',
'We preserved native frames rather than the checkpoint processor’s resize/center-crop default. That spatial-preprocessing choice has not been isolated in this audit. Nor have we completed grouped cross-validation, random-initialization controls, per-patch spatial generalization, or a full faithful optimizer replication. Do not tune toward an expected curve or present the initial result as a settled contradiction of the paper.']
(OUT/'AUDIT.md').write_text('\n'.join(lines)+'\n')
provenance={'script_sha256':hashlib.sha256((ROOT/'scripts/diagnose_reproduction.py').read_bytes()).hexdigest(),'results_sha256':hashlib.sha256((OUT/'audit_results.json').read_bytes()).hexdigest(),'slurm_job':1759839}
(OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print(OUT/'AUDIT.md')
