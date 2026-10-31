"""Scientific report for faithful pooled centroid-spline predictor transfer."""
from misc.paths import source_path
import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from misc.centroid_velocity import ROOT,SIZES
from misc.spline_velocity import sha
from misc.velocity_causal import save_json


def interval(v,scale=1.):
    return f'{scale*v["mean"]:.3f} [{scale*v["ci95"][0]:.3f}, {scale*v["ci95"][1]:.3f}]'


def plots(out,report,weak):
    d=report['splits']['D']; sizes=[s for s in SIZES if s]
    fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    for key,label,color in [('from_original','Centroid vs original','#1976b5'),
                            ('from_source_replacement','Centroid vs source-speed replacement','#1f8b66')]:
        rows=[d['conditions'][f'centroid_{s:+.2f}'][key] for s in sizes]
        gain=[r['predicted']['velocity_slope_gain'] for r in rows]
        means=np.array([v['mean'] for v in gain]); lo=np.array([v['ci95'][0] for v in gain]); hi=np.array([v['ci95'][1] for v in gain])
        axes[0,0].errorbar(sizes,means,yerr=np.maximum([means-lo,hi-means],0),fmt='o-',color=color,label=label,capsize=3)
        axes[0,1].plot(sizes,[r['context_last_shift_pixels']['mean'] for r in rows],'o-',color=color,label=label)
    linear=[d['conditions'][f'linear_{s:+.2f}'] for s in sizes]
    axes[0,0].plot(sizes,[r['predicted']['velocity_slope_gain']['mean'] for r in linear],'o-',color='#d77720',label='Iterated linear vs original')
    random=[d['conditions'][f'random_{s:+.2f}']['from_source_replacement'] for s in sizes]
    axes[0,0].plot(sizes,[r['predicted']['velocity_slope_gain']['mean'] for r in random],'.:',color='gray',label='Rotated control vs its source replacement')
    axes[0,0].axhline(1,color='black',linestyle='--');axes[0,0].axhline(0,color='gray',linewidth=.6)
    axes[0,0].set(title='Velocity-like response through predictor',ylabel='Velocity slope gain')
    axes[0,0].legend(fontsize=7)
    axes[0,1].plot(sizes,[r['context_last_shift_pixels']['mean'] for r in linear],'o-',color='#d77720',label='Iterated linear')
    axes[0,1].axhline(d['source_replacement']['centroid']['context_last_shift_pixels']['mean'],color='#1976b5',linestyle='--',label='Source replacement alone')
    axes[0,1].set(title='Independent context-position drift',ylabel='Position change (px)');axes[0,1].legend(fontsize=7)
    rows=[d['conditions'][f'centroid_{s:+.2f}'] for s in sizes]
    axes[1,0].plot(sizes,[r['relative_edit_norm']['mean'] for r in rows],'o-',color='#1976b5',label='Full centroid replacement')
    axes[1,0].plot(sizes,[r['source_target_relative_edit_norm']['mean'] for r in rows],'o-',color='#1f8b66',label='Source-to-target difference')
    axes[1,0].plot(sizes,[r['relative_edit_norm']['mean'] for r in linear],'o-',color='#d77720',label='Iterated linear')
    axes[1,0].set(title='Edit magnitudes',ylabel='Edit norm / raw activation norm'); axes[1,0].legend(fontsize=7)
    axes[1,1].plot(sizes,[r['pooled_speed_change_gain']['mean'] for r in rows],'o-',color='#1976b5')
    axes[1,1].axhline(1,color='black',linestyle='--')
    axes[1,1].set(title='Independent pooled speed readout',ylabel='Decoded speed change / requested change')
    for ax in axes.flat:
        ax.set_xlabel('Requested speed change (m/s)');ax.set_xticks(sizes);ax.grid(alpha=.2)
    fig.savefig(out/'comparison.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,path in zip(axes,weak['D']['paths']):
        xx=np.linspace(path['start'],path['end'],21)
        for method,color in [('spline','#1976b5'),('chord','#d77720')]:
            ax.plot(xx,path[method]['mae_by_waypoint'],'o-',markersize=3,color=color,label=method)
        ax.set(title=f'{path["start"]:g} → {path["end"]:g} m/s',xlabel='Target speed (m/s)',ylabel='Pooled speed-probe target MAE (m/s)')
        ax.legend();ax.grid(alpha=.2)
    fig.savefig(out/'pooled_path_check.png',dpi=170);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results/centroid_velocity_v1');args=p.parse_args();out=args.output
    report=json.loads((out/'effects.json').read_text());weak=json.loads((out/'pooled_path_diagnostic.json').read_text())
    selection=json.loads((out/'selection.json').read_text());d=report['splits']['D'];selected=selection['selected']
    plots(out,report,weak)
    row=d['conditions']['centroid_+0.50'];source=d['source_replacement']['centroid'];linear=d['conditions']['linear_+0.50']
    gain=row['from_source_replacement']['predicted']['velocity_slope_gain']
    source_speed_bias=.5*(row['from_original']['predicted']['velocity_slope_gain']['mean']-gain['mean'])
    lines=['# Original centroid-spline recipe: prefix-token predictor test','',
           'This run restores the original empirical-centroid interpolation and PCA-coordinate replacement, using the notebook helpers unchanged. '
           'The necessary adaptations are prefix-only features and a uniform token lift of the pooled edit. '
           'It does not use the previous conditional regression spline.','',
           f'On the pooled path diagnostic, spline target MAE is **{weak["D"]["spline_interior_target_mae"]:.4f} m/s** '
           f'versus **{weak["D"]["chord_interior_target_mae"]:.4f} m/s** for endpoint chords. '
           f'For a +0.50 m/s change through the predictor, the source-to-target centroid response is '
           f'**{gain["mean"]*100:.2f}%** of the expected physical velocity response '
           f'(95% CI {gain["ci95"][0]*100:.2f}–{gain["ci95"][1]*100:.2f}%). '
           f'The previously completed iterated linear edit gives **{linear["predicted"]["velocity_slope_gain"]["mean"]*100:.2f}%** from the original representation.','',
           f'Replacing PCA coordinates at the **original speed alone** moves the independent context-position readout by '
           f'**{source["context_last_shift_pixels"]["mean"]:.3f} px**, and future last-horizon position by '
           f'**{source["future_last_shift_pixels"]["mean"]:.3f} px**. '
           f'It also shifts the decoded velocity along the original direction by approximately **{source_speed_bias:+.3f} m/s**. '
           'This non-identity source replacement is why both reference points are reported below. '
           'For positive requested changes, the total response from the untouched representation is actually negative; '
           'the positive source-to-target effect does not undo the slowdown introduced by replacement.','',
           '## Recipe and token mapping','',
           '1. Mean-pool the 1,024 layer-19 tokens from frames 0–7 only. Future frames never enter encoder attention.',
           '2. Fit 64-dimensional PCA on A. For each of 8/12/16 uniformly spaced speed anchors, average A coordinates in nearest-anchor bins.',
           '3. Interpolate the centroids exactly with a natural cubic spline using `utils.fit_centroid_spline`. Select anchor count by B reconstruction MSE at known speed.',
           '4. Replace the source pooled PCA coordinates with the target curve point, retaining the pooled orthogonal complement using the original `utils.replace_pca_coordinates` operation.',
           '5. Add that pooled change identically to every token. This exactly implements the pooled edit, preserves each token’s deviation from the mean, and minimizes total squared token edit norm among lifts with the same mean change.',
           '6. Continue the frozen encoder and predictor; apply unchanged C-trained position readouts.','',
           f'Selected **{selected["knots"]} anchors**, B reconstruction MSE {selected["validation_mse"]:.6f}; '
           f'the 64 PCs explain {100*selection["pca_explained_variance_fraction"]:.2f}% of A pooled variance. '
           f'Bin counts: {selection["bin_counts"]}.','',
           'Unlike the prior conditional experiment, there is no ridge shrinkage of the spline, no direction/position regression, '
           'no extra position-null projection, and no replacement of the curve rule by a residual-difference edit. '
           'The curve is parameterized by speed only, as in the old speed experiment. Direction and position preservation are evaluated rather than assumed.','',
           '## Source-speed replacement is a required control','',
           'Let the original pooled PCA coordinates be z. The actual target edit is `(f(s + Δs) − z) @ PCA_components`, '
           'broadcast to all tokens. At Δs = 0, `(f(s) − z)` is generally nonzero. '
           'Subtracting source-replacement outcomes measures movement along the centroid curve after that replacement. '
           'It does not equal running a residual-difference edit through the nonlinear predictor from the untouched representation.','',
           'The untouched zero-edit pass reproduces the previous baseline. Random controls apply one fixed seeded orthogonal '
           'rotation in the same 64D PCA subspace to both source and target edits. This matches individual edit norms and '
           'source-to-target difference norms while retaining the identical uniform token-lift structure. '
           'Controls are new for this mapping and differ from the earlier per-token position-preserving random directions.','',
           '## Downstream motion','',
           'All rows use the same 184 D clips and requested velocity changes along each clip’s original direction. '
           'Gain is the time slope of decoded position change with a free intercept, projected onto the requested velocity change. '
           'One is the expected physical response; zero is no aligned velocity-like response. '
           'Confidence intervals are 1,000 paired clip bootstraps conditional on the fitted models.','',
           '| Δspeed | Centroid vs original gain [95% CI] | Centroid vs source replacement gain [95% CI] | Rotated-control source-to-target gain | Iterated linear gain |',
           '|---:|---|---|---:|---:|']
    for size in [s for s in SIZES if s]:
        a=d['conditions'][f'centroid_{size:+.2f}'];r=d['conditions'][f'random_{size:+.2f}'];l=d['conditions'][f'linear_{size:+.2f}']
        lines.append(f'| {size:+.2f} | {interval(a["from_original"]["predicted"]["velocity_slope_gain"])} | '
                     f'{interval(a["from_source_replacement"]["predicted"]["velocity_slope_gain"])} | '
                     f'{r["from_source_replacement"]["predicted"]["velocity_slope_gain"]["mean"]:.4f} | '
                     f'{l["predicted"]["velocity_slope_gain"]["mean"]:.4f} |')
    lines+=['','| Δspeed | Context drift vs original (px) | Context drift vs source replacement (px) | Full relative edit norm | Source-target relative edit norm | Pooled speed-change gain |',
            '|---:|---:|---:|---:|---:|---:|']
    for size in [s for s in SIZES if s]:
        a=d['conditions'][f'centroid_{size:+.2f}']
        lines.append(f'| {size:+.2f} | {a["from_original"]["context_last_shift_pixels"]["mean"]:.3f} | '
                     f'{a["from_source_replacement"]["context_last_shift_pixels"]["mean"]:.3f} | '
                     f'{a["relative_edit_norm"]["mean"]:.5f} | {a["source_target_relative_edit_norm"]["mean"]:.5f} | '
                     f'{a["pooled_speed_change_gain"]["mean"]:.3f} |')
    lines+=['','## Independent pooled probe check','',
            'A new scalar speed ridge is fitted only on C, with alpha selected by five-fold CV within C. '
            'It is used because the old evaluator was trained on full-clip pooled features. '
            'It does not select the spline or tune downstream effects. The path diagnostic restores the old speed targets '
            '0.25, 2.125, and 4 m/s, all three unordered pairs, and 21 waypoints. Spline and chord share endpoints. '
            'Endpoint chords are evaluated only for this reproduction of the old pooled test; no tangent baseline is added.','',
            f'On unedited prefix-pooled D features, the independent speed readout has MAE **{weak["D"]["clean_mae"]:.4f} m/s**, '
            f'R² **{weak["D"]["clean_r2"]:.4f}**. The path result above averages interior-waypoint errors across the three paths.','',
            '## Limits of the conclusion','',
            'This is a faithful refit of the original pooled recipe on prefix-only features with one explicit minimal token mapping. '
            'It does not establish how other token mappings would behave. Preserving token deviations from the mean does not '
            'guarantee preserving the model’s position or direction representation. A good pooled speed score can coexist with '
            'poor direction-preserving downstream motion, and source replacement may substantially change the scene representation.','',
            'The linear reference preserves construction position probes; this faithful centroid replacement does not impose '
            'that additional constraint. Accordingly, compare downstream response together with drift and edit magnitude. '
            'D was historically inspected, so this remains exploratory. The physical interpretation still depends on the '
            'frozen position decoder, whose independent teacher-token transfer was weak in the first experiment. '
            'No new clips or edited videos are generated.','',
            '## Validation and artifacts','',
            'Tests verify exact agreement with the old replacement helper, empirical centroids and interpolation, '
            'preservation of token deviations, minimum-norm lifting, and norm-matched rotated controls. '
            'The run checks the achieved PCA coordinates and untouched predictor baseline.','',
            '- [Downstream comparison](comparison.png), [pooled path check](pooled_path_check.png).',
            '- `protocol.json`, `selection.json`, `model.npz`, and `evaluation_selection.json` freeze the recipe and evaluated cases.',
            '- `pooled_path_diagnostic.json` and `pooled_speed_readout.npz` preserve the independent pooled check.',
            '- `effects.json` and `conditions/*.npz` retain both reference comparisons and every per-video output.',
            '- `code/` and `code_sha256.json` preserve the implementation.','']
    (out/'REPORT.md').write_text('\n'.join(lines))
    (out/'code').mkdir(exist_ok=True)
    files=['paths.py','centroid_velocity.py','test_centroid_velocity.py','report_centroid_velocity.py','run_centroid_velocity.sh',
           'utils.py','velocity_causal.py','velocity_intervention.py','iterated_velocity.py','spline_velocity.py','probe_each_token.py']
    for name in files:shutil.copy2(source_path(name),out/'code'/name)
    save_json(out/'code_sha256.json',{name:sha(source_path(name)) for name in files})
    print(out/'REPORT.md',flush=True)


if __name__=='__main__':main()
