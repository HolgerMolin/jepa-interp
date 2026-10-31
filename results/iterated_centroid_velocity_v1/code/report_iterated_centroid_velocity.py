"""Report validation-stopped iterative centroid-spline residualization."""
import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from iterated_centroid_velocity import ROOT,SIZES
from spline_velocity import sha
from velocity_causal import save_json


def interval(v,scale=1.):
    return f'{scale*v["mean"]:.3f} [{scale*v["ci95"][0]:.3f}, {scale*v["ci95"][1]:.3f}]'


def plot(out,selection,effects,audit):
    caps=selection['evaluated_prefixes'];d=effects['splits']['D']
    fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    colors=['#125a9c','#61a9da','#ee9840','#b84c16']
    for size,color in zip(SIZES,colors):
        rows=[d[f'spline_k{k:03d}_{size:+.2f}'] for k in caps]
        gain=[r['predicted']['velocity_slope_gain'] for r in rows]
        mean=np.array([v['mean'] for v in gain]);lo=np.array([v['ci95'][0] for v in gain]);hi=np.array([v['ci95'][1] for v in gain])
        axes[0,0].errorbar(caps,mean,yerr=np.maximum([mean-lo,hi-mean],0),color=color,fmt='o-',capsize=3,label=f'{size:+.2f} m/s')
        axes[0,0].plot(caps,[d[f'random_k{k:03d}_{size:+.2f}']['predicted']['velocity_slope_gain']['mean'] for k in caps],':',color=color)
        axes[0,1].plot(caps,[r['context_last_shift_pixels']['mean'] for r in rows],'o-',color=color)
        axes[1,0].plot(caps,[r['relative_edit_norm']['mean'] for r in rows],'o-',color=color)
        axes[1,1].plot(caps,[r['pooled_speed_change_gain']['mean'] for r in rows],'o-',color=color)
    axes[0,0].axhline(d['linear_+0.50']['predicted']['velocity_slope_gain']['mean'],color='black',linestyle='--',label='Iterated linear (+0.50)')
    axes[0,0].axhline(0,color='gray',linewidth=.6)
    axes[0,0].set(title='Predictor velocity response (dotted: random)',ylabel='Velocity slope gain (physical expectation = 1)')
    axes[0,0].legend(fontsize=7)
    axes[0,1].set(title='Independent context-position drift',ylabel='Last-context position shift (px)')
    axes[1,0].set(title='Intervention magnitude',ylabel='Edit norm / raw activation norm')
    axes[1,1].set(title='Frozen pooled speed readout',ylabel='Decoded speed change / requested change')
    for ax in axes.flat:
        ax.set_xlabel('Number of fitted centroid curves');ax.set_xticks(caps);ax.grid(alpha=.2)
        if selection['selected_prefix']>0:ax.axvline(selection['selected_prefix'],color='gray',alpha=.4,linestyle=':')
    fig.savefig(out/'steering.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    h=selection['history'];rounds=[0]+[r['round'] for r in h]
    errors=[selection['initial_B_residual_mse']]+[r['B_residual_mse'] for r in h]
    axes[0].plot(rounds,errors,'o-',color='#125a9c');axes[0].axvline(selection['selected_prefix'],color='black',linestyle='--',label='B-selected prefix')
    axes[0].set(xlabel='Residualization rounds',ylabel='B residual MSE in original feature space',title='Validation reconstruction');axes[0].legend(fontsize=8)
    caps=[int(k) for k in audit]
    for kind,color in [('linear','#125a9c'),('rbf','#b84c16')]:
        axes[1].plot(caps,[audit[str(k)][kind]['D']['r2'] for k in caps],'o-',color=color,label=kind)
    axes[1].axhline(0,color='gray',linewidth=.6);axes[1].set(xlabel='Residualization rounds',ylabel='D speed R²',title='Fresh probes on residuals');axes[1].legend()
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(out/'residuals.png',dpi=170);plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'results/iterated_centroid_velocity_v1');args=parser.parse_args();out=args.output
    selection=json.loads((out/'selection.json').read_text());effects=json.loads((out/'effects.json').read_text());audit=json.loads((out/'residual_audit.json').read_text())
    plot(out,selection,effects,audit)
    caps=selection['evaluated_prefixes'];d=effects['splits']['D'];chosen=selection['selected_prefix'];final=caps[-1]
    first=d['spline_k001_+0.50'];last=d[f'spline_k{final:03d}_+0.50'];linear=d['linear_+0.50']
    lines=['# Iterated centroid-spline residualization','',
           f'The fixed validation rule stopped after **{selection["fitted_rounds"]} rounds** ({selection["stop_reason"]}); '
           f'the lowest B residual MSE selected prefix **{chosen}**. '
           f'For +0.50 m/s, the one-round difference edit gives **{100*first["predicted"]["velocity_slope_gain"]["mean"]:.2f}%** '
           f'of the physical velocity response, and the final attempted prefix gives '
           f'**{100*last["predicted"]["velocity_slope_gain"]["mean"]:.2f}%**, '
           f'versus **{100*linear["predicted"]["velocity_slope_gain"]["mean"]:.2f}%** for the frozen iterated linear method.','',
           'All spline conditions here use source-to-target differences added to the untouched representation. '
           'There is no initial replacement by the source-speed centroid. This isolates iteration from the replacement effect observed in the previous experiment.','',
           f'After one subtraction, fresh linear speed-probe D R² falls to **{audit["1"]["linear"]["D"]["r2"]:.4f}**, '
           f'while the nonlinear RBF probe retains **{audit["1"]["rbf"]["D"]["r2"]:.4f}**. '
           f'After {final} rounds, RBF R² remains **{audit[str(final)]["rbf"]["D"]["r2"]:.4f}**. '
           'Thus the residual transformation leaves substantial nonlinearly decodable speed information that these additional centroid fits do not remove.','',
           '## Fixed sequence construction','',
           'Each round fits 64-dimensional PCA to A residuals, computes speed-bin centroids, and interpolates them with '
           'the original natural cubic helper. B chooses among 8/12/16 anchors by reconstruction MSE in that round’s PCA coordinates. '
           'The fitted contribution is mapped back to the original 1,024-dimensional pooled feature space, then centered by its A average. '
           'Subtracting it preserves the A mean but is not an orthogonal projection. The next PCA is fitted anew to the resulting residuals.','',
           'The maximum is 16 rounds. Stop after three rounds without reducing full-feature B residual MSE by at least '
           '0.1% of the original B MSE relative to the last meaningful best. The selected prefix minimizes B residual MSE '
           'among all attempted prefixes, including round zero. Predictor evaluation includes powers of two up to the attempted count, '
           'plus the final attempted and selected positive prefixes. Attempted prefixes that fail validation are explicitly exploratory diagnostics.','',
           'No predictor outcome selects a round count. Residual-probe scores are diagnostic and do not select the spline sequence. '
           'A/B fit and selection are isolated from C/D. D has been inspected historically, so this remains exploratory.','',
           '| Round | Anchors | Full-feature B residual MSE | Reduction from preceding round | Meaningful improvement |',
           '|---:|---:|---:|---:|---|']
    lines.append(f'| 0 | — | {selection["initial_B_residual_mse"]:.7f} | — | — |')
    for r in selection['history']:
        lines.append(f'| {r["round"]} | {r["selected"]["knots"]} | {r["B_residual_mse"]:.7f} | {r["B_round_reduction"]:+.7f} | {r["meaningful_validation_improvement"]} |')
    lines+=['','## Steering','',
            'The edit is `H′ = H + Σ_j [f_j(s + Δs) − f_j(s)]`, applied identically to each prefix token. '
            'Each fitted curve is expressed in original activation coordinates. Zero requested change is an exact identity. '
            'The one-round edit is numerically verified against the previous original centroid curve’s difference. '
            'Every condition uses the same remaining encoder, pretrained predictor, C-trained position readouts, '
            'and 184 speed-restricted D clips as the preceding experiments.','',
            'The control rotates the cumulative difference with one fixed seeded orthogonal matrix in the full 1,024D channel space. '
            'It preserves the edit norm and the uniform token mapping. There is no additional position-null projection. '
            'The velocity metric is the free-intercept time slope of decoded position change projected onto the requested change. '
            'Intervals are 1,000 paired clip bootstraps, conditional on fitted models and without correction for the sweep.','',
            'Representative +0.50 m/s condition:', '',
            '| Curves | Velocity gain [95% CI] | Paired gain above one round [95% CI] | Paired gain above random [95% CI] | Context drift (px) | Relative edit norm | Pooled speed gain |',
            '|---:|---|---|---|---:|---:|---:|']
    for cap in caps:
        row=d[f'spline_k{cap:03d}_+0.50']
        lines.append(f'| {cap} | {interval(row["predicted"]["velocity_slope_gain"])} | '
                     f'{interval(row["paired_vs_one_round"]["velocity_slope_gain"])} | {interval(row["paired_vs_random"]["velocity_slope_gain"])} | '
                     f'{row["context_last_shift_pixels"]["mean"]:.3f} | {row["relative_edit_norm"]["mean"]:.5f} | {row["pooled_speed_change_gain"]["mean"]:.3f} |')
    lines+=['','| Δspeed | One-round gain | Final attempted gain | Paired final-minus-one gain [95% CI] | Iterated linear gain |',
            '|---:|---:|---:|---|---:|']
    for size in SIZES:
        a=d[f'spline_k001_{size:+.2f}'];b=d[f'spline_k{final:03d}_{size:+.2f}'];l=d[f'linear_{size:+.2f}']
        lines.append(f'| {size:+.2f} | {a["predicted"]["velocity_slope_gain"]["mean"]:.4f} | '
                     f'{b["predicted"]["velocity_slope_gain"]["mean"]:.4f} | {interval(b["paired_vs_one_round"]["velocity_slope_gain"])} | '
                     f'{l["predicted"]["velocity_slope_gain"]["mean"]:.4f} |')
    lines+=['','## Residual speed information','',
            'Fresh linear and RBF-kernel ridge probes read the pooled residuals. A fits them, B chooses ridge alpha and RBF width, '
            'and all settings are saved before D scoring. RBF widths are fixed multiples of the median A squared distance. '
            'A-centered kernels and targets include an unpenalized intercept. The audit uses all 256 D clips, '
            'whereas interventions retain the established 184-clip speed-restricted subset.','',
            '**Known speed is used to construct each residual**, including on D. This is supervised residualization, '
            'not a label-free deployable erasure method. Decodability measures what these probe families can recover '
            'after that transformation; it is not a proof that all information has or has not been causally removed.','',
            '| Removed rounds | Linear D R² | Linear D MAE (m/s) | RBF D R² | RBF D MAE (m/s) |',
            '|---:|---:|---:|---:|---:|']
    for cap,rows in audit.items():
        a,b=rows['linear']['D'],rows['rbf']['D']
        lines.append(f'| {cap} | {a["r2"]:.4f} | {a["mae_mps"]:.4f} | {b["r2"]:.4f} | {b["mae_mps"]:.4f} |')
    lines+=['','## Interpretation limits','',
            'Later curves need not correspond to independent copies of speed. They can recover variation previously outside PCA, '
            'correct earlier approximation errors, or fit noise. Validation stopping and the residual audit help distinguish useful '
            'progress from a training-only reduction, but cannot establish a nonlinear nullspace.','',
            'This experiment still operates on pooled features and a uniform token lift. It cannot directly coordinate token-specific '
            'changes discarded by pooling. Positive predictor effects must be interpreted alongside position drift and perpendicular '
            'responses. The frozen position readout remains the measurement bottleneck; teacher-token transfer was weak in the original experiment.','',
            '## Checks and artifacts','',
            'Tests cover telescoping residual subtraction, preservation of the A mean, zero/reversed differences, '
            'validation-prefix selection, and centered kernel prediction against an independent implementation. '
            'The run verifies first-round equality with the original centroid curve and zero-edit predictor equivalence.','',
            '- [Steering sweep](steering.png), [validation and residual audit](residuals.png).',
            '- `protocol.json`, `selection.json`, `models/`, `fit_checks.json`, and `evaluation_selection.json` freeze construction choices.',
            '- `effects.json` and `conditions/` retain B/D effects and per-video outputs.',
            '- `audit_selection.json`, `residual_audit.json`, and `audit_predictions.npz` retain fresh-probe diagnostics.',
            '- `code/` and `code_sha256.json` preserve implementation sources.','']
    (out/'REPORT.md').write_text('\n'.join(lines));(out/'code').mkdir(exist_ok=True)
    files=['iterated_centroid_velocity.py','test_iterated_centroid_velocity.py','report_iterated_centroid_velocity.py',
           'run_iterated_centroid_velocity.sh','centroid_velocity.py','utils.py','velocity_causal.py','velocity_intervention.py',
           'iterated_velocity.py','spline_velocity.py','probe_each_token.py']
    for name in files:shutil.copy2(ROOT/name,out/'code'/name)
    save_json(out/'code_sha256.json',{name:sha(ROOT/name) for name in files})
    print(out/'REPORT.md',flush=True)


if __name__=='__main__':main()
