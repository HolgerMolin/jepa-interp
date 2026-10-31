"""Report residual diagnosis and conditional covariance steering together."""
from misc.paths import source_path
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from misc.covariance_velocity import OUT,BASE,ROOT,PREVIOUS,SIZES
from misc.diagnose_spline_residuals import OUT as DIAG
from misc.velocity_causal import save_json
from misc.spline_velocity import sha


def read(path):return json.loads(path.read_text())
def ci(value,scale=1):
    return f'{value["mean"]*scale:.2f} [{value["ci95"][0]*scale:.2f}, {value["ci95"][1]*scale:.2f}]'


def main():
    scores=read(DIAG/'scores.json');selection=read(OUT/'selection.json');effects=read(OUT/'effects.json')
    whitening=read(OUT/'whitening_audit.json');poly=read(OUT/'span_quadratic_scores.json')
    bestpoly=read(OUT/'span_quadratic_selection.json')['selected_dim'];selected=selection['selected'];d=effects['D']
    matched=read(OUT/'norm_control/effects.json')['D']
    sqsel=read(DIAG/'selection.json')['selected'];checks=read(DIAG/'checks.json')
    text=['# Residual speed diagnostics and covariance steering','',
          f'Nonlinear speed information survives label-independent removal of the mean curve’s span. '
          f'A regularized covariance transport increases +0.50 m/s steering gain from '
          f'{100*d["mean_+0.50"]["predicted"]["velocity_slope_gain"]["mean"]:.2f}% to '
          f'{100*d["combined_+0.50"]["predicted"]["velocity_slope_gain"]["mean"]:.2f}%, '
          f'but also increases edit size and position drift. At the same per-clip edit norm as mean-only steering, '
          f'its gain is {100*matched["0.5"]["predicted"]["velocity_slope_gain"]["mean"]:.2f}%. '
          f'Frozen iterated linear steering remains stronger at {100*d["linear_+0.50"]["predicted"]["velocity_slope_gain"]["mean"]:.2f}%.','',
          'These experiments use the existing clips and GPU allocation 1887909. No clips were generated. '
          'A fits models; B selects diagnostic hyperparameters and the covariance model. D has been inspected in this research sequence, '
          'including while motivating the covariance experiment, so all conclusions are exploratory. No D predictor outcome tunes this model.','',
          '## What survives mean subtraction?','',
          'The first centroid subtraction removes linear readability, but a fixed projection removing the entire spline span also leaves '
          f'nonlinear speed readability: D R² **{scores["curve_span_removed"]["rbf"]["D"]["r2"]:.3f}**. '
          f'This projection removes {checks["curve_span_rank"]} dimensions and uses no per-clip speed labels. '
          'Consequently, at least part of the nonlinear signal genuinely exists outside the mean curve’s span. '
          'This demonstrates decodability, not causal control.','',
          'RBF on residual magnitude alone is weak; normalizing every residual to unit length retains most nonlinear readability. '
          'Standardized pairwise products of PCA coordinates recover much more than coordinate squares alone. '
          'These results support feature interactions as an important source of the signal; they do not imply that a Gaussian covariance model captures all of it.','',
          '| Representation / probe input | Linear D R² | RBF D R² |','|---|---:|---:|']
    for name in ['original','true_speed_residual','estimated_speed_residual','curve_span_removed','residual_norm','residual_unit','span_removed_norm','span_removed_unit']:
        row=scores[name];text.append(f'| {name} | {row["linear"]["D"]["r2"]:.4f} | {row["rbf"]["D"]["r2"]:.4f} |')
    text+=['',f'The estimated-speed subtraction uses the frozen C-trained speed ridge, so it needs no true label at inference. '
           'Its speed estimate is clipped to the spline support. The fixed span projection is the stronger check because it does not estimate speed either.','',
           '| Moment features | B-selected PCA dimension | D R² |','|---|---:|---:|']
    for kind in ['squares','quadratic']:
        name=sqsel[kind+'_selected']['variant'];text.append(f'| Residual {kind} plus linear coordinates | {name.split("_")[-1]} | {scores[name]["linear"]["D"]["r2"]:.4f} |')
    text.append(f'| Span-removed quadratic plus linear coordinates | {bestpoly} | {poly[str(bestpoly)]["D"]["r2"]:.4f} |')
    text+=['','## Can label-dependent subtraction create signal?','',
           'Yes. For three fixed random seeds, independently permute feature donors within A, B and D, destroying the original feature/assigned-speed pairing. '
           'Fit fresh probes before and after subtracting the curve evaluated at the assigned speed. '
           'The shuffled baseline is near chance, but the subtraction creates a predictable assigned-speed component. '
           'This is a demonstration of possible injection, not an estimate of what fraction of the real residual signal is injected. '
           'It explains why the earlier R²≈0.91 residual result alone was insufficient; the label-free span test above resolves that ambiguity in part.','',
           '| Shuffle seed index | Before subtraction RBF D R² | After subtraction RBF D R² |','|---:|---:|---:|']
    for seed in range(3):text.append(f'| {seed} | {scores[f"shuffled_{seed}"]["rbf"]["D"]["r2"]:.4f} | {scores[f"shuffled_minus_curve_{seed}"]["rbf"]["D"]["r2"]:.4f} |')
    text+=['','## Conditional covariance transport','',
           'Fit 64D PCA on A centroid residuals and standardize coordinates using A variances. '
           'Estimate speed-dependent second moments with Gaussian speed weights over A clips and shrink toward the global identity. '
           'These are moments around the original spline mean; remaining conditional mean error can contribute. '
           'B selects dimension from 8/16/32/64, bandwidth from 0.35/0.7/1.4 m/s and identity shrinkage from 0.25/0.5/0.75/1. '
           'Selection minimizes Gaussian negative log likelihood over the same 64 standardized dimensions, with the unmodified tail modeled as identity.','',
           f'Selected: **{selected["dim"]} dimensions, bandwidth {selected["bandwidth"]} m/s, shrinkage {selected["shrinkage"]}**. '
           f'B NLL is {selected["B_nll"]:.3f}, versus {selection["B_identity_nll"]:.3f} for identity. '
           f'D NLL is {read(OUT/"likelihood_audit.json")["D_selected_nll"]:.3f}, versus {read(OUT/"likelihood_audit.json")["D_identity_nll"]:.3f}. '
           'Lower is better; common Gaussian constants are omitted.','',
           'For each clip, subtract the source mean, whiten the selected residual coordinates using the source covariance, '
           'color them using the target covariance, then add the target mean. Preserve the other coordinates. '
           'Map the pooled delta identically to all tokens and continue the original encoder/predictor. '
           'Zero change is identity; reversing source and target recovers the original representation to numerical precision. '
           'The shape-only condition applies only the residual transform. The random control rotates the complete edit using the previous fixed orthogonal matrix.','',
           'All downstream cases match the previous experiment: 184 D clips, ±0.25/±0.50 m/s in the original direction, '
           'frozen predictor and C-trained position readouts. Velocity gain is the free-intercept slope of position change versus future time, '
           'projected onto the physical requested velocity. It is not a fraction of encoded causal information.','',
           '| Requested Δspeed | Mean-only gain (%) | Mean + covariance gain (%) [95% CI] | Shape-only gain (%) | Linear gain (%) | Paired improvement over mean (percentage points) [95% CI] |',
           '|---:|---:|---|---:|---:|---|']
    for size in SIZES:
        row=d[f'combined_{size:+.2f}'];text.append(f'| {size:+.2f} | {100*d[f"mean_{size:+.2f}"]["predicted"]["velocity_slope_gain"]["mean"]:.2f} | '
             f'{ci(row["predicted"]["velocity_slope_gain"],100)} | {100*d[f"shape_{size:+.2f}"]["predicted"]["velocity_slope_gain"]["mean"]:.2f} | '
             f'{100*d[f"linear_{size:+.2f}"]["predicted"]["velocity_slope_gain"]["mean"]:.2f} | {ci(row["paired_vs_mean"]["velocity_slope_gain"],100)} |')
    text+=['','Representative +0.50 m/s condition:','',
           '| Method | Velocity gain (%) [95% CI] | Context drift (px) | Relative edit norm | Perpendicular slope (px/s) |','|---|---|---:|---:|---:|']
    for kind in ['mean','combined','shape','random','linear']:
        row=d[kind+'_+0.50'];text.append(f'| {kind} | {ci(row["predicted"]["velocity_slope_gain"],100)} | {row["context_last_shift_pixels"]["mean"]:.3f} | '
              f'{row["relative_edit_norm"]["mean"]:.5f} | {row["perpendicular_slope_pixels_per_second"]["mean"]:.3f} |')
    text+=['','## Equal edit-size follow-up','',
           'After observing that covariance edits were larger, fix a follow-up control that rescales each combined delta to the '
           'norm of its mean-only delta. The rescaled edit is a dose control and no longer promises exact transport to the target moments. '
           'No scale is chosen from predictor performance. This follow-up is exploratory, and its full protocol and predictions are in `norm_control/`.','',
           '| Δspeed | Mean-only gain (%) | Equal-norm covariance gain (%) [95% CI] | Paired gain above mean (percentage points) [95% CI] | Context drift (px) |',
           '|---:|---:|---|---|---:|']
    for size in SIZES:
        row=matched[str(size)];text.append(f'| {size:+.2f} | {100*d[f"mean_{size:+.2f}"]["predicted"]["velocity_slope_gain"]["mean"]:.2f} | '
            f'{ci(row["predicted"]["velocity_slope_gain"],100)} | {ci(row["paired_vs_mean"]["velocity_slope_gain"],100)} | {row["context_last_shift_pixels"]["mean"]:.3f} |')
    text+=['','At +0.50 m/s, the equal-norm comparison shows no clear improvement, while position drift remains higher. '
           'Only the −0.50 m/s equal-norm paired interval excludes zero among these four conditions, with a small benefit. '
           'Thus covariance transport captures some useful structure but has not produced a substantial or consistent efficiency improvement over the mean edit. '
           'Iterated linear steering remains the stronger tested intervention.','']
    text+=['','Intervals use 1,000 clip bootstraps conditional on fitted models, without correction for multiple conditions. '
           'The complete B/D results, teacher-readout crosscheck, paired random and linear comparisons are in `effects.json`. '
           'The teacher readout’s previously poor transfer limits physical interpretation: these outcomes concern the frozen encoder–predictor–position-readout system, not generated videos.','',
           '## Does conditional whitening remove readability?','',
           '| Representation | Linear D R² | RBF D R² |','|---|---:|---:|']
    for name,row in whitening.items():text.append(f'| {name} | {row["linear"]["D"]["r2"]:.4f} | {row["rbf"]["D"]["r2"]:.4f} |')
    text+=['','Whitening uses the true source speed. This is a supervised diagnostic, not a label-free independence guarantee. '
           'Neither better likelihood nor reduced probe readability alone establishes better causal steering. '
           'This experiment tests one regularized covariance transport and one uniform token mapping; it cannot rule out other nonlinear steering methods.','',
           '## Validation and artifacts','',
           'Three unit tests verify noncommuting covariance transport, inverse/identity behavior, likelihood comparability across dimensions, '
           'positive definite shrinkage, and invariance of span projection to curve-span shifts. '
           'Runtime checks verify real-data round trips, exact zero-edit identity, pooled lift accuracy, and random norm matching. '
           'The A/B-selected settings are saved before D scoring. Protocols, predictions and source snapshots are retained.','',
           '![Diagnostics and steering](overview.png)','']
    (OUT/'REPORT.md').write_text('\n'.join(text))
    (DIAG/'REPORT.md').write_text('# Spline residual diagnostics\n\nSee the combined [diagnostic and steering report](../covariance_velocity_v1/REPORT.md).\n\nFull diagnostic predictions, A/B-selected settings, scores and checks are stored here.\n')
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    names=['Original','Mean-subtracted','Span removed','Norm only','Unit residual','Span quadratic']
    values=[scores[n]['rbf']['D']['r2'] for n in ['original','true_speed_residual','curve_span_removed','residual_norm','residual_unit']]+[poly[str(bestpoly)]['D']['r2']]
    axes[0].barh(names[::-1],values[::-1],color=['#3388aa']*6);axes[0].set(xlabel='Evaluation speed R²',title='Nonlinear readability (last: quadratic ridge)',xlim=(0,1))
    for kind,color in [('mean','#3388aa'),('combined','#cc6622'),('shape','#aa8866'),('linear','#333333')]:
        stats=[d[f'{kind}_{s:+.2f}']['predicted']['velocity_slope_gain'] for s in SIZES]
        vals=np.array([v['mean'] for v in stats])*100;lower=np.array([v['ci95'][0] for v in stats])*100;upper=np.array([v['ci95'][1] for v in stats])*100
        axes[1].errorbar(SIZES,vals,yerr=np.maximum([vals-lower,upper-vals],0),fmt='o-',label=kind,color=color,capsize=3)
    axes[1].axhline(0,color='gray',linewidth=.5);axes[1].set(xlabel='Requested speed change (m/s)',ylabel='Predictor velocity gain (%)',title='Steering through the predictor');axes[1].legend();axes[1].grid(alpha=.2)
    fig.savefig(OUT/'overview.png',dpi=170);plt.close(fig)
    files=['paths.py','diagnose_spline_residuals.py','covariance_velocity.py','test_covariance_velocity.py','audit_covariance_velocity.py','covariance_norm_control.py','report_spline_residual_diagnostics.py','run_covariance_velocity.sh']
    code=OUT/'code';code.mkdir(exist_ok=True)
    for name in files:shutil.copy2(source_path(name),code/name)
    save_json(OUT/'code_sha256.json',{name:sha(code/name) for name in files})
    # Confirm shape and finite predictions for every evaluated condition.
    count=0
    for path in (OUT/'conditions').glob('*.npz'):
        zz=np.load(path);assert len(zz['indices'])==368
        for key in ['predicted','teacher','context']:assert zz[key].shape==(368,4,2) and np.isfinite(zz[key]).all()
        count+=1
    assert count==12
    matched_count=0
    for path in (OUT/'norm_control').glob('matched_*.npz'):
        zz=np.load(path);assert len(zz['indices'])==368
        for key in ['predicted','teacher','context']:assert zz[key].shape==(368,4,2) and np.isfinite(zz[key]).all()
        size=float(path.stem.split('_')[1]);ref=np.load(PREVIOUS/'conditions'/f'spline_k001_{size:+.2f}.npz')
        np.testing.assert_allclose(zz['edit_norm'],ref['edit_norm'],atol=1e-9)
        matched_count+=1
    assert matched_count==4
    save_json(OUT/'artifact_checks.json',{'conditions':count,'norm_control_conditions':matched_count,'B_clips':184,'D_clips':184,'all_predictions_finite':True,
             'source_matches_fit':sha(ROOT / 'misc' / 'covariance_velocity.py')==read(OUT/'protocol.json')['source_sha256'],
             'diagnostic_source_matches':sha(ROOT / 'misc' / 'diagnose_spline_residuals.py')==read(DIAG/'protocol.json')['source_sha256']})


if __name__=='__main__':main()
