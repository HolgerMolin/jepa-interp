"""Prefix-only V-JEPA velocity intervention and frozen-predictor position assay.

Run stages on a GPU compute allocation; all output/cache paths are explicit.
Original A/B/C/D video partitions and supplied data are preserved.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from utils import load_model, load_video

ROOT = Path(__file__).resolve().parent
CHECKPOINT = Path('/scratch/users/hmolin/jepa-interp/checkpoints/vjepa2-vitl-fpc64-256')
ALPHAS = np.logspace(-4, 8, 13)
PREFIX = 8
GRID = 16
CHANNELS = 1024
TOKENS = 1024
SEED = 20261031


def save_json(path, value):
    temp = Path(str(path) + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(temp, path)


def source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def records():
    rows = json.loads((ROOT / 'results/splits.json').read_text())['records']
    rows = [r for r in rows if r['task'] == 'speed']
    assert len(rows) == 1536
    assert all(r['metadata']['motion'] == 'velocity' for r in rows)
    return rows


def pixels_to_input(frames, device='cuda'):
    x = torch.as_tensor(frames, device=device).permute(0, 1, 4, 2, 3).float() / 255
    mean = x.new_tensor([.485, .456, .406])[None, None, :, None, None]
    std = x.new_tensor([.229, .224, .225])[None, None, :, None, None]
    return (x - mean) / std


def disk_positions(frames):
    """Measure disk centroids in image coordinates, independently of network labels."""
    # The supplied MP4s decode to a bright orange disk despite the DATA.md color
    # description. Color-independent foreground avoids relying on that mismatch.
    mask = frames.max(axis=-1) > 80
    mass = mask.sum(axis=(-2, -1))
    if np.any(mass < 20):
        raise ValueError(f'Disk missing or too small: pixel counts {mass.tolist()}')
    yy, xx = np.indices(frames.shape[1:3])
    return np.stack(((mask * xx).sum(axis=(-2, -1)) / mass,
                     (mask * yy).sum(axis=(-2, -1)) / mass), axis=-1)


def spatial_features(tokens):
    """Retain four times and a 4x4 spatial grid, averaging only within each cell."""
    b = len(tokens)
    return tokens.reshape(b, 4, 4, 4, 4, 4, CHANNELS).mean(dim=(3, 5)).reshape(b, 4, -1)


@torch.inference_mode()
def encode_prefix(model, pixels, stop=19):
    # Slice BEFORE patch embedding: future pixels cannot enter any attention block.
    h = model.encoder.embeddings(pixels[:, :PREFIX])
    assert h.shape[1:] == (TOKENS, CHANNELS)
    for block in model.encoder.layer[:stop]:
        h = block(h)[0]
    return h


@torch.inference_mode()
def continue_predict(model, h):
    for block in model.encoder.layer[19:]:
        h = block(h)[0]
    context = model.encoder.layernorm(h)
    b = len(h)
    cm = [torch.arange(TOKENS, device=h.device)[None].expand(b, -1)]
    tm = [torch.arange(TOKENS, 2 * TOKENS, device=h.device)[None].expand(b, -1)]
    future = model.predictor(context, context_mask=cm, target_mask=tm).last_hidden_state
    return spatial_features(future), spatial_features(context)


def configure():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def extract(args):
    args.output.mkdir(parents=True, exist_ok=False)
    args.cache.mkdir(parents=True, exist_ok=False)
    rows = records()
    n = len(rows)
    definition = {
        'created_source_sha256': source_hash(), 'dataset': 'speed, all 1536 constant-velocity clips',
        'prefix_frames': [0, 7], 'future_frames': [8, 15], 'fps': 24,
        'intervention_block': 19, 'tokens': 1024, 'token_grid': [4, 16, 16],
        'encoder': 'prefix pixels only; float32; no future tokens enter encoder attention',
        'predictor': 'frozen pretrained predictor; default learned mask index 1; target indices 1024:2048',
        'position_features': '4x4 spatial means, independently at each of four future tubelets',
        'position_labels': 'mean measured image centroid across the two frames in each tubelet',
        'position_readout': 'ridge, C only, 5-fold CV grouped by clip; one shared future-position readout across horizons; per-horizon diagnostics also saved',
        'baseline_gate': 'on B: predicted-token position readout beats last-position baseline on mean horizon error and at the farthest horizon',
        'teacher_crosscheck': 'same protocol on clean full-clip encoder future tokens; transfer to predicted tokens',
        'A': 'velocity/position edit probes', 'B': 'probe regularization, token reliability, baseline gate',
        'C': 'independent position evaluators', 'D': 'fixed final evaluation; historically inspected dataset',
        'future_time_from_boundary_seconds': ((np.arange(8, 16).reshape(4, 2).mean(1) - 7) / 24).tolist(),
        'checkpoint': json.loads((ROOT / 'checkpoint_manifest.json').read_text()),
        'split_sha256': hashlib.sha256((ROOT / 'results/splits.json').read_bytes()).hexdigest(),
        'seed': SEED, 'cache': str(args.cache),
    }
    save_json(args.output / 'protocol.json', definition)
    save_json(args.output / 'records.json', rows)
    shapes = {'layer19': (n, TOKENS, CHANNELS),
              'predicted': (n, 4, 16 * CHANNELS),
              'teacher': (n, 4, 16 * CHANNELS),
              'context': (n, 4, 16 * CHANNELS)}
    arrays = {key: np.lib.format.open_memmap(args.cache / f'{key}.npy', mode='w+',
                                          dtype=np.float32, shape=shape)
              for key, shape in shapes.items()}
    position = np.empty((n, 16, 2), np.float64)
    model = load_model(str(CHECKPOINT))
    model.eval()
    started = time.monotonic()
    checks = {}
    for start in range(0, n, args.batch_size):
        batch_rows = rows[start:start + args.batch_size]
        frames = np.stack([load_video(ROOT / 'vjepa-physics-takehome-4E00/data/speed/videos'
                                     / f'scene_{r["id"]:04d}/video.mp4') for r in batch_rows])
        end = start + len(frames)
        for j, clip in enumerate(frames):
            position[start + j] = disk_positions(clip)
        with torch.inference_mode():
            pixels = pixels_to_input(frames)
            h = encode_prefix(model, pixels)
            pred, context = continue_predict(model, h)
            teacher = model.encoder(pixels).last_hidden_state[:, TOKENS:]
            arrays['layer19'][start:end] = h.cpu().numpy()
            arrays['predicted'][start:end] = pred.cpu().numpy()
            arrays['context'][start:end] = context.cpu().numpy()
            arrays['teacher'][start:end] = spatial_features(teacher).cpu().numpy()
            if start == 0:
                # Real-model checks: future replacement cannot alter prefix or predictor output.
                corrupted = pixels.clone()
                corrupted[:, PREFIX:] = torch.randn_like(corrupted[:, PREFIX:])
                other = encode_prefix(model, corrupted)
                torch.testing.assert_close(h, other, rtol=0, atol=0)
                other_pred, _ = continue_predict(model, other)
                torch.testing.assert_close(pred, other_pred, rtol=0, atol=0)
                standard = model.encoder(pixels[:, :PREFIX]).last_hidden_state
                torch.testing.assert_close(context, spatial_features(standard), atol=2e-5, rtol=2e-5)
                checks = {'future_pixel_replacement_max_difference': float((h - other).abs().max()),
                          'future_pixel_replacement_prediction_max_difference': float((pred-other_pred).abs().max()),
                          'prefix_continuation_matches_standard_encoder': True,
                          'gpu': torch.cuda.get_device_name(), 'predictor_shape': list(pred.shape)}
                save_json(args.output / 'extraction_checks.json', checks)
        if start == 0 or end % 32 == 0 or end == n:
            elapsed = time.monotonic() - started
            print(f'extract {end}/{n}, {elapsed/60:.1f}m elapsed, '
                  f'{elapsed/end*(n-end)/60:.1f}m remaining', flush=True)
            save_json(args.output / 'progress.json', {'stage': 'extract', 'completed': end, 'total': n})
    for value in arrays.values():
        value.flush()
    metadata = [r['metadata'] for r in rows]
    theta = np.deg2rad([m['theta_degrees'] for m in metadata])
    speed = np.array([m['speed_mps'] for m in metadata])
    velocity = speed[:, None] * np.column_stack((np.cos(theta), np.sin(theta)))
    split = np.array([r['split'] for r in rows])
    start_world = np.array([m['start_position_xy_m'] for m in metadata])
    # Calibrate meters to image pixels on A only; inspect residuals, do not assume renderer scaling.
    world = start_world[:, None] + np.arange(16)[None, :, None] / 24 * velocity[:, None]
    design = np.column_stack((world[split == 'A'].reshape(-1, 2), np.ones((sum(split == 'A')*16))))
    mapping = np.linalg.lstsq(design, position[split == 'A'].reshape(-1, 2), rcond=None)[0]
    mapped = np.concatenate((world, np.ones((*world.shape[:2], 1))), axis=-1) @ mapping
    residual = np.linalg.norm(mapped-position, axis=-1)
    np.savez(args.output / 'labels.npz', ids=np.array([r['id'] for r in rows]), split=split,
             positions=position, velocity=velocity, speed=speed, world_to_pixel=mapping,
             target_position=position[:, 8:].reshape(n, 4, 2, 2).mean(2))
    save_json(args.output / 'geometry_check.json', {'world_to_pixel': mapping.tolist(),
              'pixel_residual_mean': float(residual.mean()), 'pixel_residual_max': float(residual.max()),
              'prefix_constant_velocity_fit_mean_residual': float(np.linalg.norm(
                  position[:, :8] - (position[:, :8].mean(1)[:, None]
                  + (np.arange(8)-3.5)[None, :, None]/24 * (velocity @ mapping[:2])[:, None]), axis=-1).mean())})
    save_json(args.output / 'progress.json', {'stage': 'extraction_complete', 'completed': n, 'total': n,
                                             'seconds': time.monotonic()-started})


def ridge_path(x, y, train, evaluate, alphas=ALPHAS):
    """Exact dual ridge, float64, with training-only centering."""
    xtrain = np.asarray(x[train], dtype=np.float64)
    mean = xtrain.mean(0)
    xtrain -= mean
    ymean = y[train].mean(0)
    gram = xtrain @ xtrain.T
    values, vectors = np.linalg.eigh(gram)
    values = np.maximum(values, 0)
    rhs = vectors.T @ (y[train]-ymean)
    test = np.asarray(x[evaluate], dtype=np.float64) - mean
    kernel = test @ xtrain.T
    preds = []
    for alpha in alphas:
        dual = vectors @ (rhs / (values[:, None] + alpha))
        preds.append(kernel @ dual + ymean)
    return np.stack(preds)


def fit_readout(x, y, train, groups=None):
    errors = np.zeros(len(ALPHAS))
    if groups is None:
        folds = [(train[a], train[b]) for a, b in KFold(5, shuffle=True, random_state=SEED).split(train)]
    else:
        unique = np.unique(groups[train])
        folds = [(train[np.isin(groups[train], unique[a])], train[np.isin(groups[train], unique[b])])
                 for a, b in KFold(5, shuffle=True, random_state=SEED).split(unique)]
    for fit, val in folds:
        prediction = ridge_path(x, y, fit, val)
        errors += ((prediction-y[val])**2).mean(axis=(1, 2))
    alpha = ALPHAS[errors.argmin()]
    xa = np.asarray(x[train], dtype=np.float64)
    mean = xa.mean(0)
    xa -= mean
    ym = y[train].mean(0)
    dual = np.linalg.solve(xa @ xa.T + alpha*np.eye(len(xa)), y[train]-ym)
    coef = xa.T @ dual
    return coef, ym - mean @ coef, float(alpha), errors / 5


def position_metrics(prediction, target):
    distance = np.linalg.norm(prediction-target, axis=-1)
    return {'mean_euclidean_pixels': distance.mean(0).tolist(),
            'rmse_pixels': np.sqrt(((prediction-target)**2).sum(-1).mean(0)).tolist(),
            'r2_xy': [r2_score(target[:, t], prediction[:, t], multioutput='raw_values').tolist()
                      for t in range(4)]}


def baseline(args):
    save_json(args.output/'baseline_protocol.json', {
        'source_sha256': source_hash(),
        'primary_position_readout': 'single ridge shared across all future horizons; fit only C; 5-fold CV groups all horizons of a clip',
        'primary_training_features': 'unedited frozen-predictor future tokens, spatially pooled to 4x4',
        'crosscheck': 'separate single shared readout fitted to true future encoder tokens on C, transferred unchanged to predictor outputs',
        'per_horizon_readouts': 'diagnostic only, not used for primary intervention',
        'baseline_gate': 'B mean horizon error and farthest-horizon error both below last-observed-position baseline',
        'reason_for_shared_readout': 'prevent separate horizon-specific readouts from implementing extrapolation themselves',
    })
    labels = np.load(args.output / 'labels.npz')
    split, target = labels['split'], labels['target_position']
    c = np.flatnonzero(split == 'C')
    arrays = {key: np.load(args.cache / f'{key}.npy', mmap_mode='r')
              for key in ('predicted', 'teacher', 'context')}
    predictions = {}
    settings = {}
    weights = {}
    for key in ('predicted', 'teacher', 'context'):
        # Context predicts its observed tubelet positions: separate current-position check.
        y = (labels['positions'][:, :8].reshape(len(split), 4, 2, 2).mean(2)
             if key == 'context' else target)
        predictions[key] = np.empty_like(y)
        weights[key] = []
        settings[key] = []
        for t in range(4):
            coef, intercept, alpha, cv = fit_readout(arrays[key][:, t], y[:, t], c)
            predictions[key][:, t] = arrays[key][:, t] @ coef + intercept
            weights[key].append((coef, intercept))
            settings[key].append({'alpha': alpha, 'C_cv_mse': cv.tolist(),
                                  'alpha_grid_edge': alpha in (ALPHAS[0], ALPHAS[-1])})
            print(f'{key} position readout horizon {t}: alpha={alpha}', flush=True)
        suffix = '' if key == 'context' else '_per_horizon'
        np.savez(args.output / f'{key}{suffix}_position_readout.npz',
                 coef=np.stack([w[0] for w in weights[key]]),
                 intercept=np.stack([w[1] for w in weights[key]]),
                 alpha=np.array([s['alpha'] for s in settings[key]]), train_ids=labels['ids'][c])
    # A separate readout per horizon could itself extrapolate a copied velocity
    # representation. The primary assay instead uses ONE decoder at every future
    # time, with all tubelets from a clip grouped together in C cross-validation.
    for key in ('predicted', 'teacher'):
        predictions[f'{key}_per_horizon'] = predictions[key].copy()
        flat = arrays[key].reshape(-1, arrays[key].shape[-1])
        groups = np.repeat(np.arange(len(split)), 4)
        train = np.flatnonzero(np.repeat(split == 'C', 4))
        coef, intercept, alpha, cv = fit_readout(flat, target.reshape(-1, 2), train, groups=groups)
        predictions[key] = (flat @ coef + intercept).reshape(target.shape)
        weights[key] = [(coef, intercept)] * 4
        settings[f'{key}_shared'] = {'alpha': alpha, 'C_grouped_cv_mse': cv.tolist(),
                                     'alpha_grid_edge': alpha in (ALPHAS[0], ALPHAS[-1])}
        np.savez(args.output / f'{key}_position_readout.npz', coef=np.repeat(coef[None], 4, axis=0),
                 intercept=np.repeat(intercept[None], 4, axis=0), alpha=alpha, train_ids=labels['ids'][c],
                 shared_across_horizons=True)
        print(f'{key} SHARED position readout: alpha={alpha}', flush=True)
    predictions['teacher_on_predicted'] = np.stack([
        arrays['predicted'][:, t] @ weights['teacher'][t][0] + weights['teacher'][t][1]
        for t in range(4)], axis=1)
    predictions['last_position'] = np.repeat(labels['positions'][:, 7:8], 4, axis=1)
    times = (np.arange(8, 16).reshape(4, 2).mean(1)-7)/24
    # Baseline velocity estimated from visible positions only.
    visible_times = (np.arange(8)-3.5)/24
    fitted_velocity = np.einsum('t,ntc->nc', visible_times, labels['positions'][:, :8]) / (visible_times**2).sum()
    predictions['visible_linear_extrapolation'] = labels['positions'][:, 7:8] + times[None, :, None]*fitted_velocity[:, None]
    report = {'readout_settings': settings, 'alphas': ALPHAS.tolist(), 'splits': {}}
    # Freeze fitting settings before any D measurement.
    save_json(args.output / 'readout_selection.json', report)
    for part in ('B', 'D'):
        keep = split == part
        report['splits'][part] = {key: position_metrics(p[keep], target[keep])
                                  for key, p in predictions.items() if key != 'context'}
    p = np.array(report['splits']['B']['predicted']['mean_euclidean_pixels'])
    static = np.array(report['splits']['B']['last_position']['mean_euclidean_pixels'])
    report['baseline_gate_passed'] = bool(p.mean() < static.mean() and p[-1] < static[-1])
    report['gate_details'] = {'B_prediction_error': p.tolist(), 'B_last_position_error': static.tolist()}
    save_json(args.output / 'baseline.json', report)
    np.savez(args.output / 'baseline_predictions.npz', **predictions)
    print(json.dumps(report['gate_details']), flush=True)
    print(f'BASELINE GATE: {report["baseline_gate_passed"]}', flush=True)
    plot_baseline(args, labels, predictions, report)


def plot_baseline(args, labels, predictions, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
    times = (np.arange(8, 16).reshape(4, 2).mean(1)-7)/24
    for key in ('predicted', 'teacher', 'teacher_on_predicted', 'last_position', 'visible_linear_extrapolation'):
        ax.plot(times, report['splits']['D'][key]['mean_euclidean_pixels'], 'o-', label=key.replace('_', ' '))
    ax.set(xlabel='Seconds after last observed frame', ylabel='Mean position error (pixels)',
           title='Frozen predictor: held-out future position')
    ax.legend(fontsize=8)
    fig.savefig(args.output / 'baseline.png', dpi=160)
    plt.close(fig)
    chosen = np.flatnonzero(labels['split']=='D')[::32][:8]
    fig, axes = plt.subplots(2, 4, figsize=(13, 7), layout='constrained')
    for ax, i in zip(axes.flat, chosen):
        p = labels['positions'][i]
        ax.plot(p[:8, 0], p[:8, 1], 'k.-', label='Observed')
        ax.plot(p[8:, 0], p[8:, 1], 'g.-', label='True future')
        pred = predictions['predicted'][i]
        ax.plot(pred[:, 0], pred[:, 1], 'ro-', label='Predicted future')
        ax.set(title=f'Clip {labels["ids"][i]}', xlim=(0, 256), ylim=(256, 0), aspect='equal')
    axes[0, 0].legend(fontsize=7)
    fig.savefig(args.output / 'baseline_trajectories.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['extract', 'baseline'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=2)
    args = parser.parse_args()
    configure()
    globals()[args.stage](args)


if __name__ == '__main__':
    main()
