"""Compare frozen pooled-PCA spline replacement with source-offset preservation.

Run on a compute node. No fitting, model inference, or new clips are required.
Historical artifacts are read-only; a fresh output directory is mandatory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import socket

import numpy as np
from scipy.interpolate import PPoly

from utils import target_errors


ROOT = Path(__file__).resolve().parent
TASKS = ('direction', 'speed', 'acceleration')


def encoded(value, task):
    if task == 'direction':
        angle = np.deg2rad(value)
        return np.array([np.sin(angle), np.cos(angle)])
    return np.array([value])


def offset_edit(features, components, source_coordinates, target_coordinates):
    return features + (target_coordinates - source_coordinates) @ components


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/spline_offset_v1')
    parser.add_argument('--features', type=Path,
                        default=Path('/scratch/users/hmolin/jepa-interp/pooled_features'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    historical = json.loads((ROOT / 'results/spline/scores.json').read_text())
    steering = json.loads((ROOT / 'results/steering/scores.json').read_text())
    original_protocol = json.loads((ROOT / 'results/spline/protocol.json').read_text())
    protocol = {
        'block': 19, 'pca_dimensions': 64,
        'intervention': "h_new = h + [f(target) - f(known_source_label)] @ PCA_components",
        'source_labels': 'Known clip metadata; direction is periodic. No source-label estimation.',
        'fitting': 'None. Reuse original A-fit, B-selected PCA and spline for each task.',
        'evaluator': 'Frozen original C-trained linear readout; no refitting or recalibration.',
        'evaluation_split': 'D; previously inspected, so this is an exploratory follow-up.',
        'fixed_targets': original_protocol['targets'],
        'primary_metric': 'Mean absolute target error, equal weight per target and D clip.',
        'comparators': ['Original PCA replacement', 'Frozen 16-probe linear steering'],
        'bootstrap': {'unit': 'clip, keeping all targets together', 'paired': True,
                      'samples': 5000, 'seed': 20261031, 'interval': '95% percentile'},
        'secondary_paths': 'Same saved pairs and 21 waypoints; add the same source residual '
                           'to both spline and chord paths to preserve shared endpoints.',
        'validation': ['Reproduce all saved replacement predictions and metrics',
                       'Source-equals-target identity', 'PCA-offset and complement preservation',
                       'Direct full-space versus projected readout agreement',
                       'Periodic direction invariance'],
        'predictor': 'Not run. The existing predictor experiment already uses centroid differences.',
        'host': socket.gethostname(),
    }
    # Freeze the protocol and code before evaluating any D result.
    write_json(args.output / 'protocol.json', protocol)
    shutil.copy2(__file__, args.output / Path(__file__).name)
    records = json.loads((ROOT / 'results/splits.json').read_text())['records']
    inputs = [ROOT / 'results/splits.json', ROOT / 'results/spline/protocol.json',
              ROOT / 'results/spline/validation.json', ROOT / 'results/spline/scores.json',
              ROOT / 'results/steering/scores.json', ROOT / 'utils.py', Path(__file__)]
    scores = {}
    checks = {}
    for task_index, task in enumerate(TASKS):
        print(f'{task}: loading frozen inputs', flush=True)
        paths = {key: ROOT / path for key, path in {
            'model': f'results/spline/model_{task}.npz',
            'evaluation': f'results/spline/evaluation_{task}.npz',
            'evaluator': f'results/steering/evaluator_{task}.npz',
            'linear': f'results/steering/{task}_016_probes.npz',
        }.items()}
        paths['features'] = args.features / f'{task}.npz'
        inputs.extend(paths.values())
        saved = {}
        for key in ['model', 'evaluation', 'evaluator', 'linear']:
            with np.load(paths[key], allow_pickle=False) as archive:
                saved[key] = {k: archive[k] for k in archive.files}
        model, old, evaluator, linear = (saved[k] for k in
                                        ['model', 'evaluation', 'evaluator', 'linear'])
        # Historical scalar readouts store squeezed prediction/weight axes.
        # Use an explicit output axis throughout this comparison.
        evaluator['weights'] = np.atleast_2d(evaluator['weights'])
        if task != 'direction':
            for key in ['clean_predictions', 'fixed_target_predictions', 'predictions']:
                old[key] = old[key][..., None]
        rr = [r for r in records if r['task'] == task]
        ids = np.array([r['id'] for r in rr])
        split = np.array([r['split'] for r in rr])
        label_key = {'direction': 'theta_degrees', 'speed': 'speed_mps',
                     'acceleration': 'acceleration_mps2'}[task]
        labels = np.array([r['metadata'][label_key] for r in rr])[split == 'D']
        y = np.stack([encoded(value, task) for value in labels])
        np.testing.assert_array_equal(model['training_ids'], ids[split == 'A'])
        np.testing.assert_array_equal(model['validation_ids'], ids[split == 'B'])
        np.testing.assert_array_equal(evaluator['training_ids'], ids[split == 'C'])
        for artifact in [old, linear]:
            np.testing.assert_array_equal(artifact['test_ids'], ids[split == 'D'])
            np.testing.assert_allclose(artifact['test_targets'], y, atol=1e-12)
        with np.load(paths['features'], allow_pickle=False) as features:
            np.testing.assert_array_equal(features['ids'], ids)
            x = features['features'][split == 'D', 18].astype(np.float64)
        components = model['pca_components']
        mean = model['pca_mean']
        curve = PPoly(model['spline_coefficients'], model['spline_knots'],
                      extrapolate='periodic' if task == 'direction' else False)
        z = (x - mean) @ components.T
        source = curve(labels)
        residual = z - source
        weights, bias = evaluator['weights'], evaluator['bias']
        projected_weights = components @ weights.T
        clean = x @ weights.T + bias
        np.testing.assert_allclose(clean, old['clean_predictions'], atol=1e-10)
        np.testing.assert_allclose(components @ components.T, np.eye(64), atol=1e-10)
        identity = offset_edit(x, components, source, curve(labels))
        np.testing.assert_array_equal(identity, x)
        if task == 'direction':
            np.testing.assert_allclose(curve(labels + 360), source, atol=1e-10)
        correction = residual @ projected_weights
        norms = np.maximum(np.linalg.norm(x - mean, axis=1), 1e-12)
        old_complement = x - mean - z @ components
        predictions, errors, edit_norms = [], [], []
        replacement_predictions, replacement_errors, replacement_norms = [], [], []
        max_offset_error = max_complement_error = max_direct_error = 0.
        targets = np.array(protocol['fixed_targets'][task])
        np.testing.assert_array_equal(targets, old['fixed_targets'])
        np.testing.assert_array_equal(targets, linear['targets'])
        for target in targets:
            destination = curve(target)
            edited = offset_edit(x, components, source, destination)
            replacement = x + (destination - z) @ components
            new_z = (edited - mean) @ components.T
            np.testing.assert_allclose(new_z - destination, residual, atol=1e-9)
            complement = edited - mean - new_z @ components
            np.testing.assert_allclose(complement, old_complement, atol=1e-9)
            pred = edited @ weights.T + bias
            old_pred = replacement @ weights.T + bias
            np.testing.assert_allclose(pred, old_pred + correction, atol=1e-9)
            max_offset_error = max(max_offset_error, float(np.max(np.abs(new_z - destination - residual))))
            max_complement_error = max(max_complement_error, float(np.max(np.abs(complement - old_complement))))
            max_direct_error = max(max_direct_error, float(np.max(np.abs(pred - old_pred - correction))))
            predictions.append(pred)
            replacement_predictions.append(old_pred)
            errors.append(target_errors(pred, encoded(target, task), task))
            replacement_errors.append(target_errors(old_pred, encoded(target, task), task))
            edit_norms.append(np.linalg.norm(edited - x, axis=1) / norms)
            replacement_norms.append(np.linalg.norm(replacement - x, axis=1) / norms)
        predictions, errors, edit_norms = map(np.array, [predictions, errors, edit_norms])
        replacement_predictions, replacement_errors, replacement_norms = map(
            np.array, [replacement_predictions, replacement_errors, replacement_norms])
        np.testing.assert_allclose(replacement_predictions, old['fixed_target_predictions'], atol=1e-9)
        np.testing.assert_allclose(replacement_errors.mean(), historical[task]['mean_fixed_target_mae'], atol=1e-10)
        linear_errors = linear['target_errors']
        linear_mae = next(r['mean_evaluator_target_mae'] for r in steering[task]['sweep'] if r['probe_count'] == 16)
        np.testing.assert_allclose(linear_errors.mean(), linear_mae, atol=1e-10)
        rng = np.random.default_rng(20261031 + task_index)
        bootstrap_ids = rng.integers(len(x), size=(5000, len(x)))

        def summary(values):
            per_clip = np.asarray(values).mean(axis=0)
            interval = np.quantile(per_clip[bootstrap_ids].mean(axis=1), [.025, .975])
            return {'mean': float(per_clip.mean()), 'ci95': interval.tolist()}

        # A fixed readout is affine, so the same residual contribution can be
        # added to every saved path prediction, exactly (checked above).
        old_path = old['predictions']
        path_predictions = old_path + correction[None, None, None, :, :]
        path_errors = np.empty(path_predictions.shape[:-1])
        path_norms = np.empty_like(path_errors)
        max_path_reproduction_error = 0.
        for pair_index, path_labels in enumerate(old['path_labels']):
            for method_index in range(2):
                coordinates = old['path_coordinates'][pair_index, method_index]
                reproduced = clean[None] + (coordinates[:, None] - z[None]) @ projected_weights
                np.testing.assert_allclose(reproduced, old_path[pair_index, method_index], atol=1e-9)
                max_path_reproduction_error = max(max_path_reproduction_error,
                    float(np.max(np.abs(reproduced - old_path[pair_index, method_index]))))
                for step, label in enumerate(path_labels):
                    path_errors[pair_index, method_index, step] = target_errors(
                        path_predictions[pair_index, method_index, step], encoded(label, task), task)
                path_norms[pair_index, method_index] = np.linalg.norm(
                    coordinates[:, None] - source[None], axis=-1) / norms[None]
            np.testing.assert_allclose(path_predictions[pair_index, 0, [0, -1]],
                                       path_predictions[pair_index, 1, [0, -1]], atol=1e-9)
        task_result = {
            'n_clips': len(x), 'units': {'direction': 'degrees', 'speed': 'm/s', 'acceleration': 'm/s²'}[task],
            'replacement': {'target_mae': summary(replacement_errors),
                            'mean_relative_edit_norm': float(replacement_norms.mean())},
            'offset_preserving': {'target_mae': summary(errors),
                                  'mean_relative_edit_norm': float(edit_norms.mean())},
            'linear_16': {'target_mae': summary(linear_errors),
                          'mean_relative_edit_norm': float(linear['relative_perturbation_norms'].mean())},
            'paired_mae_change_offset_minus_replacement': summary(errors - replacement_errors),
            'paired_mae_change_offset_minus_linear16': summary(errors - linear_errors),
            'per_target': [{'target': float(t), 'replacement_mae': float(a.mean()),
                            'offset_preserving_mae': float(b.mean()), 'linear16_mae': float(c.mean())}
                           for t, a, b, c in zip(targets, replacement_errors, errors, linear_errors)],
            'preserved_offset_readout_contribution': {'mean': correction.mean(axis=0).tolist(),
                                                     'std': correction.std(axis=0).tolist()},
            'zero_requested_change': {'offset_preserving_mean_edit_norm': 0.,
                'replacement_mean_edit_norm': float(np.linalg.norm(residual @ components, axis=1).mean())},
            'paths': {name: {'replacement_interior_mae': historical[task][old_name]['interior_target_mae'],
                            'offset_preserving_interior_mae': float(path_errors[:, index, 1:-1].mean())}
                      for index, (name, old_name) in enumerate([('spline', 'spline'), ('chord', 'linear')])},
        }
        if task == 'direction':
            task_result['offset_preserving']['near_zero_vector_fraction'] = float(
                np.mean(np.linalg.norm(predictions, axis=-1) < 1e-8))
        scores[task] = task_result
        checks[task] = {'identity_exact': True, 'offset_max_abs_error': max_offset_error,
                        'complement_max_abs_error': max_complement_error,
                        'direct_vs_affine_max_abs_error': max_direct_error,
                        'historical_path_prediction_max_abs_error': max_path_reproduction_error,
                        'historical_fixed_targets_reproduced': True}
        np.savez_compressed(args.output / f'evaluation_{task}.npz',
            test_ids=ids[split == 'D'], test_targets=y, source_labels=labels, fixed_targets=targets,
            fixed_target_predictions=predictions, fixed_target_errors=errors,
            fixed_target_relative_edit_norms=edit_norms,
            replacement_predictions=replacement_predictions, replacement_errors=replacement_errors,
            preserved_offset_readout_contribution=correction,
            pairs=old['pairs'], fractions=old['fractions'], path_labels=old['path_labels'],
            path_methods=['spline', 'chord'], path_predictions=path_predictions,
            path_errors=path_errors, path_relative_edit_norms=path_norms)
        write_json(args.output / 'scores.json', scores)
        write_json(args.output / 'checks.json', checks)
        print(f"{task}: replacement={replacement_errors.mean():.6f}, "
              f"offset={errors.mean():.6f}, linear16={linear_errors.mean():.6f}", flush=True)
    write_json(args.output / 'provenance.json', {str(p): sha256(p) for p in inputs})
    print(f'Completed: {args.output}', flush=True)


if __name__ == '__main__':
    main()
