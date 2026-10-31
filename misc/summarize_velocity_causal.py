"""Create a reviewable report and paired-control analysis from saved predictions."""
from misc.paths import source_path
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from misc.velocity_causal import ROOT, save_json
from misc.velocity_intervention import effect_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    baseline = json.loads((out/'baseline.json').read_text())
    probes = json.loads((out/'edit_probe_summary.json').read_text())
    result = json.loads((out/'intervention_results.json').read_text())
    labels = np.load(out/'labels.npz')
    saved = np.load(out/'intervention_predictions.npz')
    conditions = saved['conditions'].tolist()
    indices = saved['indices']
    times = (np.arange(8, 16).reshape(4, 2).mean(1)-7)/24
    comparisons = {}
    for part in ('B', 'D'):
        mask = saved['split'] == part
        comparisons[part] = {}
        for size in (-.5, -.25, .25, .5):
            j = conditions.index(f'velocity_{size:+.2f}')
            k = conditions.index(f'random_{size:+.2f}')
            v = labels['velocity'][indices[mask]]
            expected = (size*v/np.linalg.norm(v, axis=-1, keepdims=True)) @ labels['world_to_pixel'][:2]
            comparisons[part][f'{size:+.2f}'] = effect_metrics(
                (saved['predicted'][j]-saved['predicted'][k])[mask], expected, times)
    save_json(out/'paired_control_comparison.json', comparisons)
    def ci(item):
        return f'{item["mean"]:.3f} [{item["ci95"][0]:.3f}, {item["ci95"][1]:.3f}]'
    d = result['splits']['D']
    n = d['velocity_+0.50']['videos']
    b = baseline['splits']['D']
    lines = [
        '# Velocity edits through the frozen V-JEPA predictor', '',
        'This experiment edits prefix-only layer-19 tokens, continues encoder blocks 20–24 and final LayerNorm, '
        'runs the pretrained frozen predictor, and measures future disk positions with a frozen C-trained readout.', '',
        'The primary readout is shared across all four prediction horizons. It was trained on unedited predictor outputs. '
        'A separate readout trained on real future encoder tokens is a cross-check; its poor transfer to predictor outputs '
        'limits claims about natural counterfactual representations.', '',
        '## Data and validation', '',
        '- All 1,536 constant-velocity speed-dataset clips; original A=768, B=256, C=256, D=256 partitions.',
        '- Only frames 0–7 enter the context encoder. Targets are tubelets (8,9), (10,11), (12,13), (14,15).',
        '- Replacing all future pixels with random pixels leaves prefix activations and predicted tokens exactly unchanged.',
        '- Bright-disk centroids provide image-coordinate labels. The MP4s decode to orange despite the supplied blue-disk description.',
        '- The A-calibrated world-to-image mapping is approximately (32*x+128, -32*y+128); '
        'mean metadata-to-centroid discrepancy is 0.40 pixels.',
        '- Position readouts fit C only; five-fold cross-validation groups all horizons of a video. '
        'No evaluator is trained on edits. A fits intervention probes; B selects their alpha and reliable tokens.',
        '- Baseline gate fixed before scoring: B mean-across-horizon and farthest-horizon errors must both beat holding the last position.',
        '- The existing D split has been used in earlier project experiments; this is a fixed follow-up, not a newly untouched benchmark.', '',
        '## Unedited prediction baseline', '',
        '| Readout / baseline | +0.0625 s | +0.1458 s | +0.2292 s | +0.3125 s |',
        '|---|---:|---:|---:|---:|',
    ]
    for key, name in [('predicted', 'Shared readout of predictor outputs'), ('teacher', 'Shared readout of real future tokens'),
                      ('teacher_on_predicted', 'Real-token readout transferred to predictions'),
                      ('last_position', 'Hold last observed position'),
                      ('visible_linear_extrapolation', 'Extrapolate measured prefix centroids')]:
        lines.append('| '+name+' | '+' | '.join(f'{x:.2f}' for x in b[key]['mean_euclidean_pixels'])+' |')
    lines += ['', 'Numbers are mean Euclidean position errors in pixels on D. The predictor baseline passes the predefined B gate. '
              'Simple measured-trajectory extrapolation is much more accurate; beating the static baseline does not establish superior physics prediction.', '',
              '## Intervention', '',
              f'All {probes["selected_tokens"]} prefix tokens passed B mean velocity-component R² ≥ 0.5 and '
              'mean position-component R² ≥ 0.8. Each token has a four-output ridge probe for velocity (vx,vy) '
              'and the last observed image position (px,py). Targets are standardized using A only.', '',
              'For each selected token, the minimum-norm edit satisfies W_velocity Δh = Δv and W_position Δh = 0. '
              'Velocity shifts are ±0.25 and ±0.50 m/s along the original direction. The raw probe constraint residual is '
              f'{probes["constraint_max_residual_selected"]:.2e}. This algebraic constraint does not guarantee position preservation downstream.', '',
              f'The fixed evaluation subset contains {n} D clips with speeds in [0.75,3.5] m/s, keeping edited speeds inside the training range. '
              'Random controls preserve the same token position probes and match each token’s edit norm, with a fixed random direction per token. '
              'Zero edits reproduce the cached baseline.', '',
              '## Measured effects on D', '',
              'A physically calibrated velocity edit would produce Δp(t) = (t−t₀) Δv. '
              'Velocity slope gain is the slope of the measured position change, projected onto the requested image velocity change, '
              'divided by that requested change. A gain of 1 matches the physical effect; 0 means no slope change. '
              'A free intercept separates a position offset from a velocity-like slope change.', '',
              '| Δspeed (m/s) | Velocity slope gain [95% CI] | Random slope gain [95% CI] | Paired gain difference [95% CI] | Final projected shift / expected (px) |',
              '|---|---|---|---|---|']
    for size in (-.5, -.25, .25, .5):
        v = d[f'velocity_{size:+.2f}']['predicted']
        r = d[f'random_{size:+.2f}']['predicted']
        pair = comparisons['D'][f'{size:+.2f}']['velocity_slope_gain']
        lines.append(f'| {size:+.2f} | {ci(v["velocity_slope_gain"])} | {ci(r["velocity_slope_gain"])} | {ci(pair)} | '
                     f'{v["mean_projected_displacement_pixels"][-1]:.3f} / {v["expected_projected_displacement_pixels"][-1]:.3f} |')
    lines += ['', 'Intervals use 1,000 paired clip-bootstrap resamples and condition on the fitted models. '
              'They do not include training uncertainty or correction across the four intervention sizes.', '',
              '| Δspeed (m/s) | Position-offset intercept norm (px) | Independently decoded last-context-tubelet shift (px) | Edit norm / raw activation norm |',
              '|---|---:|---:|---:|']
    for size in (-.5, -.25, .25, .5):
        item = d[f'velocity_{size:+.2f}']
        lines.append(f'| {size:+.2f} | {item["predicted"]["position_intercept_pixels"]["mean"]:.3f} | '
                     f'{item["context_last_tubelet_shift_pixels"]["mean"]:.3f} | '
                     f'{item["relative_edit_norm"]["mean"]:.4f} |')
    specific = all(comparisons['D'][f'{s:+.2f}']['velocity_slope_gain']['ci95'][0]>0 for s in (-.5, -.25, .25, .5))
    lines += ['', '## Interpretation', '',
              ('The velocity edits produce a positive velocity-like slope effect relative to matched random controls at all four edit sizes. '
               if specific else 'The tested velocity edits do not show a consistently positive velocity-like slope effect relative to matched random controls across all four sizes. ')
              + 'The table reports the effect size rather than treating a changed construction probe as success.', '',
              'The causal conclusion is restricted to this frozen encoder–predictor–position-readout pipeline. '
              'The predictor and encoder were not trained or optimized to make the edits succeed. '
              'The position decoder is calibrated on natural predictor outputs, so off-distribution decoder responses remain possible. '
              'The clean-encoder position decoder does not transfer reliably to predictor outputs and cannot validate the counterfactual positions independently. '
              'Position drift after the remaining encoder blocks is measured above; preserving a layer-19 position probe alone does not exclude that confound.', '',
              'Only a four-dimensional probe row space per token was edited. Distributed or redundant velocity information may remain outside it. '
              'A weak or null effect would therefore not prove that velocity is noncausal. '
              'The test varies speed along the original direction; direction rotation and acceleration edits have not been run.', '',
              '## Artifacts', '',
              '- [Baseline error curves](baseline.png) and [baseline trajectories](baseline_trajectories.png).',
              '- [Velocity effect versus horizon](velocity_effect.png) and [edited trajectories](edited_trajectories.png).',
              '- `baseline.json`, `edit_probe_summary.json`, `intervention_results.json`, and `paired_control_comparison.json` contain detailed metrics.',
              '- `intervention_predictions.npz` contains every evaluated video, condition, position prediction, and edit norm.',
              '- `protocol.json` records extraction; `baseline_protocol.json` refines the assay to a shared decoder before baseline fitting; '
              '`intervention_protocol.json` and `intervention_selection.json` freeze the interventions.',
              '- `code/` and `code_sha256.json` preserve the final experiment implementation.', '']
    (out/'REPORT.md').write_text('\n'.join(lines))
    (out/'code').mkdir(exist_ok=True)
    hashes = {}
    for name in ('paths.py', 'velocity_causal.py', 'velocity_intervention.py', 'summarize_velocity_causal.py',
                 'test_velocity_causal.py', 'run_velocity_causal.sh', 'run_velocity_intervention.sh',
                 'probe_each_token.py', 'utils.py'):
        shutil.copy2(source_path(name), out/'code'/name)
        hashes[name] = hashlib.sha256((source_path(name)).read_bytes()).hexdigest()
    save_json(out/'code_sha256.json', hashes)
    print(out/'REPORT.md')


if __name__ == '__main__':
    main()
