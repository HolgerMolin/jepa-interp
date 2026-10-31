"""Frozen protocol for token-wise velocity edits followed by the V-JEPA predictor.

Position evaluators are fitted on C in velocity_causal.py. Edit probes fit A,
select regularization/reliable tokens using B, and never train on interventions.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from probe_each_token import fit_token_batch
from velocity_causal import (CHECKPOINT, SEED, configure, continue_predict,
                             load_model, save_json, source_hash)


def component_r2(y, prediction):
    # prediction [video, token, output]
    residual = ((prediction-y[:, None])**2).sum(axis=0)
    total = ((y-y.mean(0))**2).sum(axis=0)
    return 1-residual/total


def constrained_operator(coef, scales):
    """coef [token, (vx,vy,px,py),channel] in physical units.

    Return [token,channel,2] mapping desired velocity changes in m/s to the
    minimum-norm activation change with zero current-position probe change.
    Target standardization only conditions the pseudoinverse; constraints are exact.
    """
    normalized = coef/scales[None, :, None]
    inverse = np.linalg.pinv(normalized, rcond=1e-8)
    return inverse[:, :, :2]/scales[None, None, :2]


def fit(args):
    baseline = json.loads((args.output/'baseline.json').read_text())
    if not baseline['baseline_gate_passed']:
        raise RuntimeError('The predefined unedited position-prediction gate failed; inspect baseline first')
    labels = np.load(args.output/'labels.npz')
    split = labels['split']
    y = np.column_stack((labels['velocity'], labels['positions'][:, 7]))
    mean, scale = y[split=='A'].mean(0), y[split=='A'].std(0)
    normalized = (y-mean)/scale
    # Explicitly prevent accidental use of C/D labels in construction.
    normalized[(split=='C') | (split=='D')] = np.nan
    x = np.load(args.cache/'layer19.npy', mmap_mode='r')
    definition = {
        'method': 'per-token joint ridge for vx,vy and last-observed px,py; A-centered, A-label-standardized',
        'alpha_selection': 'per-token B mean absolute error over the four standardized targets',
        'alphas': [1e-4, 1e-3, 1e-2, .1, 1, 10, 100, 1000, 10000],
        'reliable_tokens': 'B mean velocity-component R2 >= .5 and mean position-component R2 >= .8; numerical rank four',
        'edit': 'minimum norm shift of each selected velocity probe by delta_v, preserving its position outputs',
        'conditions_mps': [-.5, -.25, 0, .25, .5],
        'velocity_direction': 'original physical velocity direction, determined by metadata',
        'random_controls': 'fixed seeded random channel directions per token projected orthogonal to position-probe rows, matched to each token edit norm; both signs and magnitudes',
        'evaluation_videos': 'all B and D clips with original speed in [0.75,3.5] m/s; fixed before effects',
        'evaluators': 'unchanged C-trained future-position readout, teacher-position transfer, and context-position readout',
        'primary_measure': 'change of predicted image position versus horizon*delta_v transformed to pixels',
        'position_confound': 'free-intercept regression over time; report intercept and velocity slope separately',
        'uncertainty': '1000 paired clip-bootstrap resamples; conditional on fitted models and this split',
        'interpretation': 'effect on frozen predictor plus frozen position evaluator; no guaranteed natural counterfactual',
    }
    save_json(args.output/'intervention_protocol.json', definition)
    coef = np.empty((1024, 4, 1024), np.float64)
    intercept = np.empty((1024, 4), np.float64)
    val_r2 = np.empty((1024, 4), np.float64)
    alpha = np.empty(1024)
    started = time.monotonic()
    for start in range(0, 1024, args.token_batch_size):
        end = min(1024, start+args.token_batch_size)
        batch = np.asarray(x[:, start:end])
        fitted = fit_token_batch(batch, normalized, split, 'velocity')
        coef[start:end] = fitted['coef']*scale[None, :, None]
        intercept[start:end] = fitted['intercept']*scale+mean
        prediction = np.einsum('ntc,toc->nto', batch[split=='B'], coef[start:end]) + intercept[None, start:end]
        val_r2[start:end] = component_r2(y[split=='B'], prediction)
        alpha[start:end] = fitted['alpha']
        if start == 0 or end % 128 == 0 or end == 1024:
            elapsed = time.monotonic()-started
            print(f'edit probes {end}/1024, {elapsed/60:.1f}m elapsed, '
                  f'{elapsed/end*(1024-end)/60:.1f}m remaining', flush=True)
    values = np.linalg.svd(coef/scale[None, :, None], compute_uv=False)
    selected = ((val_r2[:, :2].mean(1) >= .5) & (val_r2[:, 2:].mean(1) >= .8)
                & (values[:, -1] > values[:, 0]*1e-8))
    operator = constrained_operator(coef, scale)
    achieved = np.einsum('toc,tcd->tod', coef, operator)
    expected = np.zeros((4, 2)); expected[:2] = np.eye(2)
    if selected.any():
        np.testing.assert_allclose(achieved[selected], np.broadcast_to(expected, achieved[selected].shape), atol=1e-6)
    np.savez(args.output/'edit_probes.npz', coef=coef, intercept=intercept, validation_r2=val_r2,
             alpha=alpha, selected=selected, operator=operator, target_mean=mean, target_scale=scale,
             train_ids=labels['ids'][split=='A'], validation_ids=labels['ids'][split=='B'])
    report = {'selected_tokens': int(selected.sum()), 'total_tokens': 1024,
              'validation_r2_quantiles_by_component': np.quantile(val_r2, [.1, .5, .9], axis=0).tolist(),
              'alpha_grid_edge_fraction': float(((alpha==1e-4)|(alpha==1e4)).mean()),
              'constraint_max_residual_selected': float(np.max(np.abs(achieved[selected]-expected))) if selected.any() else None,
              'seconds': time.monotonic()-started}
    save_json(args.output/'edit_probe_summary.json', report)
    print(json.dumps(report), flush=True)


def readout(features, weights):
    return np.einsum('btf,tfo->bto', features, weights['coef']) + weights['intercept']


def bootstrap_mean(values):
    rng = np.random.default_rng(SEED)
    draws = rng.integers(len(values), size=(1000, len(values)))
    return {'mean': float(np.mean(values)),
            'ci95': np.quantile(np.mean(values[draws], axis=1), [.025, .975]).tolist()}


def effect_metrics(change, expected_velocity, times):
    norm = np.linalg.norm(expected_velocity, axis=-1)
    unit = expected_velocity/norm[:, None]
    expected = times[None, :, None]*expected_velocity[:, None]
    gain = (change*expected).sum(axis=(1, 2))/(expected**2).sum(axis=(1, 2))
    tc = times-times.mean()
    slope = np.einsum('t,ntc->nc', tc, change)/(tc**2).sum()
    intercept = change.mean(1)-times.mean()*slope
    slope_gain = (slope*expected_velocity).sum(1)/(expected_velocity**2).sum(1)
    return {'displacement_gain': bootstrap_mean(gain), 'velocity_slope_gain': bootstrap_mean(slope_gain),
            'position_intercept_pixels': bootstrap_mean(np.linalg.norm(intercept, axis=1)),
            'mean_projected_displacement_pixels': np.einsum('ntc,nc->nt', change, unit).mean(0).tolist(),
            'expected_projected_displacement_pixels': (times[None]*norm[:, None]).mean(0).tolist(),
            'counterfactual_displacement_error_pixels': np.linalg.norm(change-expected, axis=-1).mean(0).tolist()}


def intervene(args):
    labels = np.load(args.output/'labels.npz')
    probes = np.load(args.output/'edit_probes.npz')
    selected = probes['selected']
    if not selected.any():
        save_json(args.output/'intervention_status.json', {'status': 'no_tokens_passed_validation_thresholds'})
        return
    x = np.load(args.cache/'layer19.npy', mmap_mode='r')
    evaluators = {k: np.load(args.output/f'{k}_position_readout.npz')
                  for k in ('predicted', 'teacher', 'context')}
    keep = np.flatnonzero(np.isin(labels['split'], ['B', 'D']) &
                         (labels['speed'] >= .75) & (labels['speed'] <= 3.5))
    # Save the final evaluation cases/conditions BEFORE running any intervention.
    conditions = [('zero', 0.)]
    for size in (-.5, -.25, .25, .5):
        conditions.extend([(f'velocity_{size:+.2f}', size), (f'random_{size:+.2f}', size)])
    save_json(args.output/'intervention_selection.json', {
        'ids': labels['ids'][keep].tolist(), 'splits': labels['split'][keep].tolist(),
        'conditions': [{'name': name, 'delta_speed_mps': size} for name, size in conditions],
        'selected_token_indices': np.flatnonzero(selected).tolist()})
    arrays = {key: np.empty((len(conditions), len(keep), 4, 2))
              for key in ('predicted', 'teacher', 'context')}
    edit_norm = np.zeros((len(conditions), len(keep)))
    source_norm = np.empty(len(keep))
    operator = torch.as_tensor(probes['operator'], dtype=torch.float32, device='cuda')
    operator[~selected] = 0
    rng = np.random.default_rng(SEED)
    random_array = rng.normal(size=(1024, 1024))
    # Match the position-preservation constraint as well as per-token edit norms.
    q = np.linalg.svd(probes['coef'][:, 2:], full_matrices=False)[2]
    random_array -= np.einsum('toc,to->tc', q, np.einsum('toc,tc->to', q, random_array))
    np.testing.assert_allclose(np.einsum('toc,tc->to', probes['coef'][:, 2:], random_array), 0, atol=1e-6)
    random = torch.as_tensor(random_array, dtype=torch.float32, device='cuda')
    random /= torch.linalg.vector_norm(random, dim=-1, keepdim=True)
    model = load_model(str(CHECKPOINT))
    started = time.monotonic()
    for j, (name, size) in enumerate(conditions):
        for start in range(0, len(keep), args.batch_size):
            ii = keep[start:start+args.batch_size]
            end = start+len(ii)
            h = torch.as_tensor(np.array(x[ii]), device='cuda')
            v = labels['velocity'][ii]
            delta = size*v/np.linalg.norm(v, axis=1, keepdims=True)
            edit = torch.einsum('tcd,bd->btc', operator, torch.as_tensor(delta, dtype=torch.float32, device='cuda'))
            if name.startswith('random'):
                edit = np.sign(size)*torch.linalg.vector_norm(edit, dim=-1, keepdim=True)*random[None]
            edit_norm[j, start:end] = torch.linalg.vector_norm(edit.flatten(1), dim=1).cpu().numpy()
            source_norm[start:end] = torch.linalg.vector_norm(h.flatten(1), dim=1).cpu().numpy()
            with torch.inference_mode():
                future, context = continue_predict(model, h+edit)
            future, context = future.cpu().numpy(), context.cpu().numpy()
            for key in ('predicted', 'teacher'):
                arrays[key][j, start:end] = readout(future, evaluators[key])
            arrays['context'][j, start:end] = readout(context, evaluators['context'])
            if start == 0 or end % 64 == 0 or end == len(keep):
                print(f'{name}: {end}/{len(keep)}, total {(time.monotonic()-started)/60:.1f}m', flush=True)
        np.savez(args.output/'intervention_progress.npz', conditions=np.array([c[0] for c in conditions[:j+1]]),
                 indices=keep, **{k: a[:j+1] for k, a in arrays.items()})
    # Verify the zero edit is the cached original computation, including all readouts.
    previous = np.load(args.output/'baseline_predictions.npz')
    for key, original_key in [('predicted', 'predicted'), ('teacher', 'teacher_on_predicted'), ('context', 'context')]:
        np.testing.assert_allclose(arrays[key][0], previous[original_key][keep], atol=2e-3, rtol=1e-5)
    np.savez(args.output/'intervention_predictions.npz', conditions=np.array([c[0] for c in conditions]),
             delta_speed=np.array([c[1] for c in conditions]), indices=keep, ids=labels['ids'][keep],
             split=labels['split'][keep], edit_norm=edit_norm, source_norm=source_norm, **arrays)
    report = {'zero_edit_matches_cached_baseline': True, 'seconds': time.monotonic()-started,
              'selected_tokens': int(selected.sum()), 'splits': {}}
    times = (np.arange(8, 16).reshape(4, 2).mean(1)-7)/24
    for part in ('B', 'D'):
        mask = labels['split'][keep] == part
        report['splits'][part] = {}
        for j, (name, size) in enumerate(conditions[1:], start=1):
            v = labels['velocity'][keep[mask]]
            delta_v = size*v/np.linalg.norm(v, axis=1, keepdims=True)
            expected_velocity = delta_v @ labels['world_to_pixel'][:2]
            item = {key: effect_metrics((a[j]-a[0])[mask], expected_velocity, times)
                    for key, a in arrays.items() if key != 'context'}
            item['context_last_tubelet_shift_pixels'] = bootstrap_mean(np.linalg.norm(
                (arrays['context'][j]-arrays['context'][0])[mask, -1], axis=-1))
            item['relative_edit_norm'] = bootstrap_mean((edit_norm[j]/source_norm)[mask])
            item['videos'] = int(mask.sum())
            report['splits'][part][name] = item
    save_json(args.output/'intervention_results.json', report)
    plot(args, report, conditions, arrays, labels, keep)
    print('Completed fixed velocity intervention and random controls', flush=True)


def plot(args, report, conditions, arrays, labels, keep):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    times = (np.arange(8, 16).reshape(4, 2).mean(1)-7)/24
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
    for ax, size in zip(axes.flat, [-.5, -.25, .25, .5]):
        for method in ('velocity', 'random'):
            item = report['splits']['D'][f'{method}_{size:+.2f}']['predicted']
            ax.plot(times, item['mean_projected_displacement_pixels'], 'o-', label=method)
        ax.plot(times, item['expected_projected_displacement_pixels'], 'k--', label='Physical expectation')
        ax.set(title=f'Delta speed {size:+.2f} m/s', xlabel='Seconds after observed boundary',
               ylabel='Displacement along requested change (px)')
        ax.axhline(0, color='gray', linewidth=.5)
        ax.legend()
    fig.savefig(args.output/'velocity_effect.png', dpi=170)
    plt.close(fig)
    d = np.flatnonzero(labels['split'][keep]=='D')[::24][:8]
    fig, axes = plt.subplots(2, 4, figsize=(13, 7), layout='constrained')
    j = [name for name, _ in conditions].index('velocity_+0.50')
    for ax, local in zip(axes.flat, d):
        i = keep[local]
        p = labels['positions'][i]
        ax.plot(p[:8, 0], p[:8, 1], 'k.-', label='Observed')
        for index, label, color in [(0, 'Original prediction', 'blue'), (j, 'Velocity +0.5 m/s', 'red')]:
            q = arrays['predicted'][index, local]
            ax.plot(q[:, 0], q[:, 1], 'o-', color=color, label=label)
        ax.set(title=f'Clip {labels["ids"][i]}', aspect='equal')
        ax.invert_yaxis()
    axes[0, 0].legend(fontsize=7)
    fig.savefig(args.output/'edited_trajectories.png', dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['fit', 'intervene'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--token-batch-size', type=int, default=16)
    args = parser.parse_args()
    configure()
    globals()[args.stage](args)


if __name__ == '__main__':
    main()
