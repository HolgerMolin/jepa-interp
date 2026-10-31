"""Plots and scientific report for the iterated token-velocity experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from velocity_causal import ROOT, save_json


def interval(item):
    return f'{item["mean"]:.3f} [{item["ci95"][0]:.3f}, {item["ci95"][1]:.3f}]'


def plot(out,summary,effects,audit,baseline_path):
    caps = summary['unique_caps']; d = effects['splits']['D']
    fig,axes = plt.subplots(2,2,figsize=(12,9),layout='constrained')
    colors = ['#125a9c','#3498db','#ee9840','#b84c16']
    for size,color in zip([-.5,-.25,.25,.5],colors):
        items = [d[f'velocity_k{k:03d}_{size:+.2f}'] for k in caps]
        random = [d[f'random_k{k:03d}_{size:+.2f}'] for k in caps]
        gain = [v['predicted']['velocity_slope_gain']['mean'] for v in items]
        lo = [v['predicted']['velocity_slope_gain']['ci95'][0] for v in items]
        hi = [v['predicted']['velocity_slope_gain']['ci95'][1] for v in items]
        axes[0,0].plot(caps,gain,'o-',color=color,label=f'{size:+.2f} m/s')
        axes[0,0].fill_between(caps,lo,hi,color=color,alpha=.12)
        axes[0,0].plot(caps,[v['predicted']['velocity_slope_gain']['mean'] for v in random],':',color=color)
        axes[0,1].plot(caps,[v['relative_edit_norm']['mean'] for v in items],'o-',color=color,label=f'{size:+.2f} m/s')
        axes[1,0].plot(caps,[v['context_last_shift_pixels']['mean'] for v in items],'o-',color=color)
        axes[1,1].plot(caps,[v['predicted']['position_intercept_pixels']['mean'] for v in items],'o-',color=color)
    axes[0,0].axhline(1,color='black',linestyle='--',label='Physical slope gain = 1')
    axes[0,0].axhline(0,color='gray',linewidth=.5)
    axes[0,0].set(title='Velocity response (dotted: matched random)',ylabel='Velocity slope gain')
    axes[0,0].legend(fontsize=8)
    axes[0,1].set(title='Intervention size',ylabel='Edit norm / raw activation norm')
    axes[1,0].set(title='Independent context-position check',ylabel='Last context tubelet position shift (pixels)')
    axes[1,1].set(title='Position offset in future predictions',ylabel='Free-intercept norm (pixels)')
    for ax in axes.flat:
        ax.set_xscale('log',base=2); ax.set_xticks(caps,labels=[str(k) for k in caps])
        ax.set_xlabel('Maximum eligible probes per token'); ax.grid(alpha=.2)
    fig.savefig(out/'steering_sweep.png',dpi=170); plt.close(fig)
    times = (np.arange(8,16).reshape(4,2).mean(1)-7)/24
    fig,axes = plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for kind,color in [('erase','#125a9c'),('random_erase','#ee9840')]:
        vals = [d[f'{kind}_k{k:03d}']['mean_position_error_pixels'][-1] for k in caps]
        axes[0].plot(caps,vals,'o-',color=color,label=kind.replace('_',' '))
        axes[1].plot(times,d[f'{kind}_all']['mean_position_error_pixels'],'o-',color=color,label=kind.replace('_',' ')+' (all discovered)')
    # Erasure uses the speed-restricted subset; obtain its exact zero-edit error.
    labels = np.load(baseline_path/'labels.npz')
    zero = np.load(out/'conditions/zero.npz'); mask = zero['split']=='D'
    error = np.linalg.norm(zero['predicted'][mask]-labels['target_position'][zero['indices'][mask]],axis=-1).mean(0)
    axes[0].axhline(error[-1],color='black',linestyle='--',label='Original')
    axes[1].plot(times,error,'k--',label='Original')
    axes[0].set_xscale('log',base=2); axes[0].set_xticks(caps,labels=[str(k) for k in caps])
    axes[0].set(xlabel='Maximum eligible probes erased per token',ylabel='Farthest-horizon position error (px)',title='Erasing reliable velocity subspaces')
    axes[1].set(xlabel='Seconds after observation boundary',ylabel='Position error (px)',title='Erasing all discovered directions')
    for ax in axes: ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.savefig(out/'erasure_sweep.png',dpi=170); plt.close(fig)
    counts = np.load(out/'token_counts.npz')
    fig,axes = plt.subplots(2,4,figsize=(13,7),layout='constrained')
    for row,(key,title) in enumerate([('eligible_counts','Eligible velocity probes'),('terminal_rank','Removed dimensions at termination')]):
        grids = counts[key].reshape(4,16,16)
        for t in range(4):
            im = axes[row,t].imshow(grids[t],vmin=0,vmax=grids.max(),cmap='viridis')
            axes[row,t].set(title=f'{title}\nObserved frames {2*t}–{2*t+1}',xlabel='Patch x',ylabel='Patch y')
        fig.colorbar(im,ax=axes[row].tolist(),shrink=.8)
    fig.savefig(out/'token_probe_maps.png',dpi=170); plt.close(fig)
    fig,ax = plt.subplots(figsize=(9,5),layout='constrained')
    keys = list(audit)
    values = np.array([audit[k]['D']['r2_components'] for k in keys])
    xx = np.arange(len(keys))
    ax.bar(xx-.18,values[:,0],.36,label='vx'); ax.bar(xx+.18,values[:,1],.36,label='vy')
    ax.set_xticks(xx,labels=[k.replace('_','\n') for k in keys]); ax.set(ylabel='Held-out joint-token R²',title='Velocity remaining after terminal erasure')
    ax.legend(); ax.axhline(0,color='gray',linewidth=.5)
    fig.savefig(out/'joint_audit.png',dpi=170); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'results/iterated_velocity_v1')
    parser.add_argument('--baseline',type=Path,default=ROOT/'results/velocity_causal_v1')
    args = parser.parse_args(); out = args.output
    summary = json.loads((out/'sequence_summary.json').read_text())
    effects = json.loads((out/'effects.json').read_text())
    audit = json.loads((out/'joint_audit.json').read_text())
    plot(out,summary,effects,audit,args.baseline)
    caps = summary['unique_caps']; last = caps[-1]; d = effects['splits']['D']
    first = d['velocity_k001_+0.50']['predicted']['velocity_slope_gain']
    full = d[f'velocity_k{last:03d}_+0.50']['predicted']['velocity_slope_gain']
    erasure = d['erase_all']['mean_position_error_pixels'][-1]
    random_erasure = d['random_erase_all']['mean_position_error_pixels'][-1]
    lines = ['# Iterated token velocity interventions', '',
             f'For the +0.50 m/s intervention, using all validation-eligible probes changes the velocity-like response '
             f'from **{100*first["mean"]:.2f}%** of the physical expectation with one probe to **{100*full["mean"]:.2f}%**. '
             f'Erasing all discovered velocity directions gives **{erasure:.2f} px** farthest-horizon position error, '
             f'versus **{random_erasure:.2f} px** for rank-matched random erasure. '
             f'However, full erasure also shifts the independent context-position readout by '
             f'**{d["erase_all"]["context_last_shift_pixels"]["mean"]:.2f} px**, so that damage cannot be attributed selectively to velocity. '
             'The joint-token audit still recovers substantial velocity information after erasure.', '',
             'This follow-up fits separate sequences of velocity probes at each of 1,024 prefix-only layer-19 tokens. '
             'It tests coordinated steering and erasure through the frozen remaining encoder, pretrained predictor, '
             'and unchanged position evaluators from the first experiment.', '',
             '## Frozen protocol', '',
             '- Data, prefix, splits, checkpoint, and C-trained position evaluators are unchanged from the first experiment.',
             '- Each token’s existing position-probe row space is protected. All velocity readouts are constrained to its orthogonal complement.',
             '- Fit each velocity probe on A; choose ridge alpha on B using mean standardized component MAE. Remove its effective row space and refit.',
             '- A probe qualifies for steering only if BOTH vx and vy have B R² ≥ 0.5 and both readout directions are independent.',
             '- Stop separately for each token after three consecutive probes with BOTH component R² ≤ 0.05 and BOTH MAE ≥ 90% of the training-mean baseline on B; '
             'also stop at numerical rank exhaustion or 256 removed velocity dimensions. Fit a final residual probe at the cap.',
             '- Probe-count sweeps specify a maximum; each token uses only its available eligible probes. Identical saturated caps are not rerun.',
             '- Random steering controls preserve position rows and match per-token edit norms. Random erasure controls preserve position rows and match per-token removed rank.',
             '- Downstream evaluation uses the same speed-restricted B/D clips and changes of ±0.25 and ±0.50 m/s along the original velocity direction.',
             '- D was inspected in earlier experiments. This is a fixed exploratory follow-up; no cap is selected using downstream D performance.', '',
             'The solver uses float64 equality-constrained ridge updates, periodically rebuilds the inverse, and verifies results against independent direct dual solves. '
             'Tests compare the full fit–project–refit sequence with explicit scikit-learn fits, verify C/D isolation, verify orthogonality and erasure, '
             'and compare the simultaneous edit with a direct stacked pseudoinverse.', '',
             '## Different tokens require different probe counts', '',
             f'Eligible-probe count quantiles (min, 10%, 25%, median, 75%, 90%, max): **{summary["eligible_probe_count_quantiles"]}**.', '',
             f'Terminal removed-rank quantiles in the same order: **{summary["terminal_rank_quantiles"]}**.', '',
             f'Stopping reasons: `{json.dumps(summary["stop_reasons"])}`. A cap stop does not establish exhaustion.', '',
             '| Maximum eligible probes | Actual probes: min / median / max | Mean edited dimensions per token |',
             '|---:|---|---:|']
    for cap in caps:
        dist = summary['cap_distributions'][str(cap)]
        lines.append(f'| {cap} | {dist["probes_min"]} / {dist["probes_median"]:g} / {dist["probes_max"]} | {dist["mean_edited_dimensions"]:.2f} |')
    lines += ['', '## Coordinated steering', '',
              'The edit shifts every included effective velocity readout by the same Δv while preserving the protected position rows. '
              'Because the sequential readout row spaces are orthogonal, summing their minimum-norm edits equals the stacked constrained solution.', '',
              'The primary measure below is velocity slope gain: a free-intercept regression of decoded position change against future time, '
              'projected onto the requested velocity change. A gain of 1 is physically calibrated; 0 is no velocity-like response. '
              'Intervals are 1,000 paired clip-bootstrap resamples, conditional on the fitted models, without correction for the full sweep.', '',
              'Representative +0.50 m/s condition (all four changes and both B/D are in `effects.json`):', '',
              '| Maximum probes | Velocity slope gain [95% CI] | Paired gain above random [95% CI] | Final projected shift (px; expected 5) | Context position drift (px) | Relative edit norm |',
              '|---:|---|---|---:|---:|---:|']
    for cap in caps:
        item = d[f'velocity_k{cap:03d}_+0.50']
        lines.append(f'| {cap} | {interval(item["predicted"]["velocity_slope_gain"])} | '
                     f'{interval(item["paired_vs_random"]["velocity_slope_gain"])} | '
                     f'{item["predicted"]["mean_projected_displacement_pixels"][-1]:.3f} | '
                     f'{item["context_last_shift_pixels"]["mean"]:.3f} | {item["relative_edit_norm"]["mean"]:.4f} |')
    lines += ['', 'The matched one-probe condition here uses the same protected-position fit as the multi-probe conditions. '
              'The first experiment used a joint four-output fit, so its 3.3% slope gain is historical context rather than an exactly matched baseline.', '',
              '## Erasure', '',
              'Erase the A-centered coordinates in each eligible velocity subspace, and separately all discovered velocity directions '
              'including probes below the steering threshold. Positive paired error difference means targeted erasure harms prediction more than random erasure.', '',
              '| Removed subspace | Farthest-horizon error (px) | Random erasure error (px) | Paired extra error [95% CI] | Context position drift (px) |',
              '|---|---:|---:|---|---:|']
    for cap in [*caps,-1]:
        suffix = 'all' if cap==-1 else f'k{cap:03d}'
        item,r = d[f'erase_{suffix}'],d[f'random_erase_{suffix}']
        lines.append(f'| {"All discovered" if cap==-1 else "Up to "+str(cap)+" eligible probes"} | '
                     f'{item["mean_position_error_pixels"][-1]:.3f} | {r["mean_position_error_pixels"][-1]:.3f} | '
                     f'{interval(item["paired_error_vs_random_pixels"][-1])} | {item["context_last_shift_pixels"]["mean"]:.3f} |')
    lines += ['', '## Joint-token residual audit', '',
              'A fresh exact dual ridge reads all 1,024×1,024 token features jointly. It fits A, selects alpha on B, and scores D. '
              'This checks information recoverable by combining tokens even when their individual residual probes are weak.', '',
              '| Representation | D R² vx | D R² vy | D MAE vx (m/s) | D MAE vy (m/s) |',
              '|---|---:|---:|---:|---:|']
    for key,item in audit.items():
        metric = item['D']
        lines.append('| '+key.replace('_',' ')+' | '+' | '.join(f'{v:.4f}' for v in metric['r2_components']+metric['mae_components_mps'])+' |')
    lines += ['', 'The position-and-velocity-erased representation is an audit diagnostic only; actual downstream erasure preserves the position directions. '
              'Residual joint decodability prevents interpreting per-token stopping as complete removal of velocity information.', '',
              '## Interpretation and limitations', '',
              'This experiment measures interventions in the frozen encoder–predictor–readout pipeline. It does not generate a changed video. '
              'Stronger velocity-like responses than the matched one-probe baseline support redundant information as one explanation of the initial weak edit. '
              'A weak response, or erasure without selective harm, would not establish that velocity is noncausal.', '',
              'Position preservation is guaranteed only for the construction readouts. Independent context-position drift and future-position offsets '
              'must be considered alongside slope gain. Larger subspace edits can create unnatural representations or disrupt unrelated properties.', '',
              'The frozen position evaluator is calibrated on unedited predictor outputs. The independently trained real-future-token position evaluator '
              'transferred poorly in the first experiment and remains an unreliable physical cross-check. '
              'Bootstrap intervals omit uncertainty in training the probes/evaluators. One seeded random basis is used per token; the controls do not average over multiple random bases.', '',
              '## Artifacts', '',
              '- [Steering sweep](steering_sweep.png), [erasure sweep](erasure_sweep.png), [token probe maps](token_probe_maps.png), [joint audit](joint_audit.png).',
              '- `protocol.json` and `evaluation_selection.json` freeze the methods and evaluated cases.',
              '- `sequence_summary.json`, `token_counts.npz`, and `probe_history.npz` describe each sequence.',
              '- `effects.json` and `conditions/*.npz` contain all effect metrics and per-video predictions.',
              '- `joint_audit.json`, `joint_audit_selection.json`, and `joint_audit_predictions.npz` contain the joint residual audit.',
              '- Large sequence weights and bases remain in the scratch directory recorded in `protocol.json`.',
              '- `code/` and `code_sha256.json` preserve this implementation.', '']
    (out/'REPORT.md').write_text('\n'.join(lines))
    (out/'code').mkdir(exist_ok=True)
    hashes = {}
    for name in ['iterated_velocity.py','evaluate_iterated_velocity.py','report_iterated_velocity.py',
                 'test_iterated_velocity.py','run_iterated_velocity.sh','run_iterated_evaluation.sh',
                 'velocity_causal.py','velocity_intervention.py','utils.py']:
        shutil.copy2(ROOT/name,out/'code'/name)
        hashes[name] = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    save_json(out/'code_sha256.json',hashes)
    print(out/'REPORT.md')


if __name__=='__main__':
    main()
