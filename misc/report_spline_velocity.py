"""Report the frozen spline-versus-iterated-linear predictor experiment."""
from misc.paths import source_path
import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from misc.spline_velocity import ROOT, SIZES, sha
from misc.velocity_causal import save_json


def interval(metric,scale=1.):
    return f'{scale*metric["mean"]:.3f} [{scale*metric["ci95"][0]:.3f}, {scale*metric["ci95"][1]:.3f}]'


def plots(out,effects):
    d=effects['splits']['D']
    colors={'spline':'#1469a7','linear':'#d77720','random':'#6b6b6b','linear_random':'#aaa67a'}
    fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    for method in ('spline','linear','random','linear_random'):
        rows=[d[f'{method}_{size:+.2f}'] for size in SIZES]
        gains=[r['predicted']['velocity_slope_gain'] for r in rows]
        values=np.array([s*g['mean'] for s,g in zip(SIZES,gains)])
        bounds=np.sort(np.array([s*np.array(g['ci95']) for s,g in zip(SIZES,gains)]),axis=1)
        style='o-' if method in ('spline','linear') else '.:'
        axes[0,0].errorbar(SIZES,values,yerr=np.maximum(np.array([values-bounds[:,0],bounds[:,1]-values]),0),
                           fmt=style,color=colors[method],label=method.replace('_',' '),capsize=3)
        axes[0,1].plot(SIZES,[r['context_last_shift_pixels']['mean'] for r in rows],style,color=colors[method])
        axes[1,0].plot(SIZES,[r['relative_edit_norm']['mean'] for r in rows],style,color=colors[method])
        axes[1,1].plot(SIZES,[r['perpendicular_slope_pixels_per_second']['mean'] for r in rows],style,color=colors[method])
    axes[0,0].plot(SIZES,SIZES,'k--',alpha=.5,label='Physical expectation')
    axes[0,0].set(title='Decoded velocity change along original direction',ylabel='Equivalent speed change (m/s)')
    axes[0,0].legend(fontsize=8)
    axes[0,1].set(title='Position preservation outside construction probes',ylabel='Last-context position drift (px)')
    axes[1,0].set(title='Intervention magnitude',ylabel='Edit norm / raw activation norm')
    axes[1,1].set(title='Unintended perpendicular response',ylabel='Perpendicular position slope (px/s)')
    for ax in axes.flat:
        ax.set_xlabel('Requested speed change (m/s)'); ax.grid(alpha=.2); ax.set_xticks(SIZES)
    fig.savefig(out/'comparison.png',dpi=170); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    times=(np.arange(8,16).reshape(4,2).mean(1)-7)/24
    for ax,size in zip(axes,[-.5,.5]):
        for method in ('spline','linear','random'):
            r=d[f'{method}_{size:+.2f}']['predicted']
            ax.plot(times,r['mean_projected_displacement_pixels'],'o-',color=colors[method],label=method)
        ax.plot(times,r['expected_projected_displacement_pixels'],'k--',label='Physical expectation')
        ax.set(title=f'{size:+.2f} m/s; projection along requested change',xlabel='Seconds after observation boundary',ylabel='Decoded position change (px)')
        ax.legend(); ax.grid(alpha=.2)
    fig.savefig(out/'future_displacement.png',dpi=170); plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'results/spline_velocity_v1')
    args=parser.parse_args(); out=args.output
    effects=json.loads((out/'effects.json').read_text())
    selection=json.loads((out/'selection.json').read_text())
    checks=json.loads((out/'fit_checks.json').read_text())
    protocol=json.loads((out/'protocol.json').read_text())
    d=effects['splits']['D']; s=d['spline_+0.50']; l=d['linear_+0.50']; random=d['random_+0.50']
    plots(out,effects)
    gain=s['predicted']['velocity_slope_gain']; lg=l['predicted']['velocity_slope_gain']
    best=selection['selected']
    lines=['# Conditional spline velocity steering','',
           f'For +0.50 m/s, spline steering produces **{gain["mean"]*100:.2f}%** of the physically expected velocity response '
           f'(95% clip-bootstrap CI {gain["ci95"][0]*100:.2f}–{gain["ci95"][1]*100:.2f}%), '
           f'versus **{lg["mean"]*100:.2f}%** for the frozen iterated linear method. '
           f'The spline-matched random control gives **{random["predicted"]["velocity_slope_gain"]["mean"]*100:.2f}%**.','',
           f'At this requested change, independent context-position drift is **{s["context_last_shift_pixels"]["mean"]:.3f} px** '
           f'for the spline and **{l["context_last_shift_pixels"]["mean"]:.3f} px** for linear steering. '
           f'Their mean edit/raw-activation norm ratios are **{s["relative_edit_norm"]["mean"]:.4f}** and '
           f'**{l["relative_edit_norm"]["mean"]:.4f}**, respectively.','',
           '## Fixed experiment','',
           'The task is to change speed while keeping direction and observation-boundary position fixed. '
           'This reuses the prefix-only layer-19 activations (frames 0–7), remaining encoder blocks, pretrained predictor, '
           'and frozen C-trained position readouts from the velocity experiments. No future frame enters the encoder input. '
           'The same speed-restricted B/D examples receive ±0.25 and ±0.50 m/s changes. '
           'The baseline is the previously completed iterated linear method using all B-eligible probes (saturated cap 32); '
           'its saved per-example outcomes are reused and case identities are checked.','',
           '## Curve construction and selection','',
           'Each of the 1,024 tokens and 1,024 channels has its own regression coefficients. The shared input basis is '
           'a natural cubic speed spline multiplied by periodic sine/cosine direction features and quadratic '
           'boundary-position features (1, x, y, x², xy, y²). Coordinates use known speed/direction and measured frame-7 position. '
           'This is a new token-level conditional spline regression; the old pooled centroid spline weights are not reused. '
           'There is no output PCA truncation.','',
           'A fits all parameters and input scaling. B chooses among 8/12/16 speed anchors, one/two direction harmonics, '
           'and the fixed ridge grid, using protected-activation reconstruction MSE. '
           'The existing position-probe row spaces are projected out of the fitted outputs. '
           'An exact output Gram matrix permits selection over all token/channel outputs without approximating them.','',
           f'Selected: **{best["knots"]} speed anchors, direction order {best["direction_order"]}, '
           f'{best["features"]} input features, ridge alpha {best["alpha"]:g}**. '
           f'B reconstruction MSE is {best["validation_mse"]:.6f}, versus '
           f'{selection["training_mean_validation_mse"]:.6f} for the A-mean representation '
           f'({100*selection["validation_fraction_explained_vs_A_mean"]:.2f}% reduction). '
           'This reconstruction score includes position/direction structure and is not a velocity-specific accuracy score.','',
           'The intervention is `H′ = H + f(s + Δs, θ, p7) − f(s, θ, p7)`. '
           'It retains the original example residual and gives exactly zero edit when Δs = 0. '
           'Direction and boundary position are held fixed in both curve evaluations; target speeds remain within the fitted domain. '
           'The edit is projected again to remove float32 position-constraint residuals. '
           'There is no fitted gain multiplier or downstream choice of edit strength.','',
           'Random controls use the same seeded position-null directions as the preceding experiment, '
           'with each token’s norm matched to its spline edit. All choices are saved before downstream D evaluation.','',
           '## Downstream comparison','',
           'Velocity slope gain is the time slope of decoded position change, with a free intercept, '
           'projected onto the requested velocity change. One means the physical response is fully recovered; '
           'zero means no aligned velocity response. It does not measure the fraction of causal information captured. '
           'Intervals use 1,000 paired clip-bootstrap resamples, conditional on the fitted models.','',
           '| Requested Δspeed | Spline gain [95% CI] | Linear gain [95% CI] | Paired spline − linear gain [95% CI] | Spline random gain |',
           '|---:|---|---|---|---:|']
    for size in SIZES:
        a,b,r=[d[f'{kind}_{size:+.2f}'] for kind in ('spline','linear','random')]
        lines.append(f'| {size:+.2f} | {interval(a["predicted"]["velocity_slope_gain"])} | '
                     f'{interval(b["predicted"]["velocity_slope_gain"])} | {interval(a["paired_vs_linear"]["velocity_slope_gain"])} | '
                     f'{r["predicted"]["velocity_slope_gain"]["mean"]:.4f} |')
    lines+=['','| Δspeed | Method | Context drift (px) | Future-position intercept norm (px) | Perpendicular slope (px/s) | Relative edit norm |',
            '|---:|---|---:|---:|---:|---:|']
    for size in SIZES:
        for kind in ('spline','linear'):
            item=d[f'{kind}_{size:+.2f}']
            lines.append(f'| {size:+.2f} | {kind} | {item["context_last_shift_pixels"]["mean"]:.3f} | '
                         f'{item["predicted"]["position_intercept_pixels"]["mean"]:.3f} | '
                         f'{item["perpendicular_slope_pixels_per_second"]["mean"]:.3f} | {item["relative_edit_norm"]["mean"]:.5f} |')
    slow=d['spline_-0.50']; slow_linear=d['linear_-0.50']
    lines+=['','## Interpretation and limits','',
            f'At −0.50 m/s the spline response is {100*slow["predicted"]["velocity_slope_gain"]["mean"]:.2f}% '
            f'versus {100*slow_linear["predicted"]["velocity_slope_gain"]["mean"]:.2f}% for linear steering. '
            f'At +0.50 m/s the paired spline-minus-linear gain is '
            f'{interval(s["paired_vs_linear"]["velocity_slope_gain"],100)} percentage points. '
            f'The +0.50 m/s spline edit has {s["relative_edit_norm"]["mean"]/l["relative_edit_norm"]["mean"]:.2f} times '
            f'the linear edit norm and {s["context_last_shift_pixels"]["mean"]/l["context_last_shift_pixels"]["mean"]:.2f} times '
            'the context-position drift. These comparisons assess this particular conditional fit, not every possible spline construction.','',
            'Assess the aligned velocity response together with position drift, perpendicular response, and edit size. '
            'A higher response with substantially larger or less selective edits is a tradeoff. '
            'This comparison evaluates the two complete steering methods; it does not separately identify the contribution of curvature.','',
            'The spline is a regularized conditional mean fitted to sparse coverage of direction, speed, and position. '
            'Conditioning and preserving the construction position probes do not guarantee a natural counterfactual or preservation '
            'of all position information. Holding boundary position fixed while changing speed also changes the implied past trajectory. '
            'Reconstruction-based selection may favor nuisance structure over the most causally useful velocity variation.','',
            'D has already been inspected in prior experiments; this is an exploratory follow-up. '
            'The position evaluator is calibrated on unedited predictor outputs, and the independent teacher-token readout '
            'transferred poorly in the original experiment. Effects therefore concern the frozen encoder–predictor–readout pipeline. '
            'The experiment does not render an edited video. Confidence intervals omit fitting uncertainty and are not corrected '
            'for the sweep. One seeded random direction is used per token.','',
            '## Verification and artifacts','',
            'Unit tests verify exact agreement with independent scikit-learn ridge fits, Gram-based validation MSE, '
            'periodicity, no extrapolation, zero/reversed edits, and projection commuting with the fitted map. '
            'The complete run independently verifies selected-model reconstruction against direct predictions, '
            'position constraints, and zero-edit agreement with the previous predictor baseline.','',
            f'- Direct B MSE: {checks["direct_validation_mse"]:.10f}; Gram B MSE: {checks["gram_validation_mse"]:.10f}.',
            f'- Float64 coefficient position-constraint residual: {checks["position_constraint_max_residual"]:.3e}.',
            '- [Comparison plot](comparison.png), [future displacement](future_displacement.png).',
            '- `protocol.json`, `selection.json`, `model.npz`, `fit_checks.json`, `evaluation_selection.json` preserve construction decisions.',
            '- `effects.json` reports B/D metrics for every method and requested change; `conditions/` stores new per-video outputs.',
            f'- Full token coefficients: `{protocol["work"]}/coefficients.npy`; checksum in `fit_checks.json`.',
            '- `code/` and `code_sha256.json` preserve source files.','']
    (out/'REPORT.md').write_text('\n'.join(lines))
    (out/'code').mkdir(exist_ok=True)
    files=['paths.py','spline_velocity.py','test_spline_velocity.py','report_spline_velocity.py','run_spline_velocity.sh',
           'velocity_causal.py','velocity_intervention.py','iterated_velocity.py','utils.py','probe_each_token.py']
    for name in files:
        shutil.copy2(source_path(name),out/'code'/name)
    save_json(out/'code_sha256.json',{name:sha(source_path(name)) for name in files})
    print(out/'REPORT.md',flush=True)


if __name__=='__main__':
    main()
