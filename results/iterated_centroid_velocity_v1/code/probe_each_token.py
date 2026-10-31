"""Independent 1024-feature ridge probes for every token at every encoder block.

768 A videos fit each probe; B chooses its alpha by MAE; C is untouched;
D evaluates it. Token positions never share weights or become extra videos.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from probe_full_tokens import ROOT, SCRATCH, TASKS, advance_layer, initial_tokens
from utils import load_data, load_features, load_model


ALPHAS = [1e-4, 1e-3, 1e-2, .1, 1., 10., 100., 1000., 10000.]
METRICS = ('r2', 'mae')


def scores(y, prediction, task):
    """y [video, output]; prediction [token, video, output]; result [token, 2]."""
    residual = (prediction - y).square().sum(dim=1)
    total = (y - y.mean(dim=0)).square().sum(dim=0)
    r2 = torch.where(total > 0, 1 - residual / total,
                     torch.where(residual == 0, 1., 0.)).mean(dim=-1)
    error = absolute_errors(y, prediction, task)
    return torch.stack((r2, error.mean(dim=-1)), dim=-1)


def absolute_errors(y, prediction, task):
    if task != 'direction':
        return (prediction - y).abs().mean(dim=-1)
    angle = torch.atan2(prediction[..., 0], prediction[..., 1])
    angle = torch.where(torch.linalg.vector_norm(prediction, dim=-1) < 1e-8, 0., angle)
    true_angle = torch.atan2(y[:, 0], y[:, 1])
    return torch.rad2deg(((angle - true_angle + torch.pi) % (2 * torch.pi) - torch.pi).abs())


@torch.inference_mode()
def fit_token_batch(features, y, split, task, alphas=ALPHAS, device='cuda'):
    """Batch independent exact ridge solves; features [video, token, channel].

    No D labels are read. Each token has its own training mean, alpha, and weights.
    Float64 dual eigensolves reuse a factorization across the original alpha grid.
    """
    x = torch.as_tensor(np.ascontiguousarray(features.transpose(1, 0, 2)),
                        dtype=torch.float64, device=device)
    a = torch.as_tensor(np.flatnonzero(split == 'A'), device=device)
    b = torch.as_tensor(np.flatnonzero(split == 'B'), device=device)
    d = torch.as_tensor(np.flatnonzero(split == 'D'), device=device)
    if len(a) < 2 or len(b) < 2 or len(d) < 2:
        raise ValueError('Each of A, B, D needs at least two videos')
    ya = torch.as_tensor(y[split == 'A'], dtype=torch.float64, device=device)
    yb = torch.as_tensor(y[split == 'B'], dtype=torch.float64, device=device)
    mean = x[:, a].mean(dim=1, keepdim=True)
    xa, xb = x[:, a] - mean, x[:, b] - mean
    y_mean = ya.mean(dim=0)
    gram = xa @ xa.transpose(1, 2)
    values, vectors = torch.linalg.eigh(gram)
    if torch.any(values[:, 0] < -1e-8 * values[:, -1].clamp_min(1.)):
        raise ValueError('Training kernel is not numerically positive semidefinite')
    values = values.clamp_min(0.)
    rhs = vectors.transpose(1, 2) @ (ya - y_mean)
    val_kernel = xb @ xa.transpose(1, 2)
    validation = []
    for alpha in alphas:
        dual = vectors @ (rhs / (values[..., None] + alpha))
        validation.append(scores(yb, val_kernel @ dual + y_mean, task))
    validation = torch.stack(validation, dim=1)  # [token, alpha, metric]
    best = validation[:, :, 1].argmin(dim=1)
    alpha = torch.as_tensor(alphas, dtype=torch.float64, device=device)[best]
    dual = vectors @ (rhs / (values[..., None] + alpha[:, None, None]))
    coef = (xa.transpose(1, 2) @ dual).transpose(1, 2)
    intercept = y_mean - (mean @ coef.transpose(1, 2)).squeeze(1)
    direct_val = x[:, b] @ coef.transpose(1, 2) + intercept[:, None]
    dual_val = val_kernel @ dual + y_mean
    torch.testing.assert_close(direct_val, dual_val, atol=1e-6, rtol=1e-5)
    token = torch.arange(len(x), device=device)
    selected = validation[token, best]
    training = scores(ya, xa @ coef.transpose(1, 2) + y_mean, task)
    result = {
        'coef': coef, 'intercept': intercept, 'alpha': alpha,
        'alpha_at_grid_edge': (best == 0) | (best == len(alphas) - 1),
        'validation_grid': validation,
        'validation_r2': selected[:, 0], 'validation_mae': selected[:, 1],
        'train_r2': training[:, 0], 'train_mae': training[:, 1],
        'test_prediction': x[:, d] @ coef.transpose(1, 2) + intercept[:, None],
        'readout_singular_values': torch.linalg.svdvals(coef),
        'primal_dual_max_difference': (direct_val - dual_val).abs().flatten(1).amax(dim=1),
    }
    return {name: value.cpu().numpy() for name, value in result.items()}


@torch.inference_mode()
def evaluate_tokens(y, predictions, task, device='cuda', repeats=500, seed=20260929):
    """Clip-level bootstrap, sharing resamples across token positions."""
    target = torch.as_tensor(y, dtype=torch.float64, device=device)
    prediction = torch.as_tensor(predictions, dtype=torch.float64, device=device)
    measured = scores(target, prediction, task)
    rng = np.random.default_rng(seed)
    draws = torch.as_tensor(rng.integers(0, len(y), size=(repeats, len(y))), device=device)
    counts = torch.zeros((repeats, len(y)), dtype=torch.float64, device=device)
    counts.scatter_add_(1, draws, torch.ones_like(draws, dtype=torch.float64))
    weights = counts / len(y)
    mae = absolute_errors(target, prediction, task) @ weights.T
    # Compute bootstrap target variances around their means to avoid cancellation.
    centered_y = target - target.mean(dim=0)
    variance = weights @ centered_y.square() - (weights @ centered_y).square()
    variance = variance.clamp_min(0.)
    mse = torch.einsum('bn,tno->tbo', weights, (prediction - target).square())
    r2 = torch.where(variance > 0, 1 - mse / variance,
                     torch.where(mse == 0, 1., 0.)).mean(dim=-1)
    quantiles = prediction.new_tensor([.025, .975])
    return {
        'test_r2': measured[:, 0].cpu().numpy(),
        'test_mae': measured[:, 1].cpu().numpy(),
        'test_r2_ci95': torch.quantile(r2, quantiles, dim=1).T.cpu().numpy(),
        'test_mae_ci95': torch.quantile(mae, quantiles, dim=1).T.cpu().numpy(),
    }


def summarize(path):
    with np.load(path) as saved:
        summary = {'layer': int(saved['layer']), 'tokens': len(saved['alpha']),
                   'seconds': float(saved['seconds']),
                   'validation_fraction_r2_ge_0_5': float((saved['validation_r2'] >= .5).mean()),
                   'alpha_grid_edge_fraction': float(saved['alpha_at_grid_edge'].mean())}
        for split in ('train', 'validation', 'test'):
            for metric in METRICS:
                values = saved[f'{split}_{metric}']
                summary[f'{split}_{metric}_quantiles'] = np.quantile(values, [.1, .5, .9]).tolist()
        return summary


def save_plot(results, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    pooled = json.loads((ROOT / 'results/probe_results.json').read_text())
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout='constrained')
    units = {'direction': 'degrees', 'speed': 'm/s', 'acceleration': 'm/s²'}
    for col, task in enumerate(TASKS):
        for row, metric in enumerate(METRICS):
            ax = axes[row, col]
            for split in ('validation', 'test'):
                quantiles = np.array([s[f'{split}_{metric}_quantiles'] for s in results[task]])
                line, = ax.plot(range(1, 25), quantiles[:, 1], label=f'Token median: {split}')
                ax.fill_between(range(1, 25), quantiles[:, 0], quantiles[:, 2],
                                color=line.get_color(), alpha=.12)
            ax.plot(range(1, 25), [s['test'][metric] for s in pooled[task]['layers']],
                    '--', color='gray', label='Historical pooled test')
            ax.set(title=task, xlabel='Block',
                   ylabel='R²' if metric == 'r2' else f'MAE ({units[task]})')
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle('Independent token probes; shading = 10th–90th percentile across tokens')
    fig.savefig(output / 'layerwise.png', dpi=160)
    plt.close(fig)


def protocol():
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        'representation': 'one independent raw 1024-channel probe per token position per block',
        'token_grid': [8, 16, 16], 'token_order': 'time, height, width',
        'layers': list(range(1, 25)), 'tasks': list(TASKS), 'total_probes': 147456,
        'alphas': ALPHAS, 'alpha_selection': 'each token independently minimizes B MAE',
        'fit': 'A', 'validation': 'B', 'reserved': 'C (not extracted)', 'test': 'D',
        'direction_target': ['sin(theta)', 'cos(theta)'],
        'scalars': {'speed': 'm/s', 'acceleration': 'm/s²'},
        'solver': 'exact dual ridge with intercept; float64 batched eigensolves',
        'feature_scaling': 'training mean centering only',
        'inference': 'frozen float32, TF32 disabled, raw residuals before final LayerNorm',
        'preprocessing': 'native 16 RGB frames at 256x256 with ImageNet normalization',
        'bootstrap_replicates': 500, 'bootstrap_seed': 20260929,
        'layer_or_token_selection': 'none; all positions saved, future thresholds must use B',
        'validation_fraction_r2_ge_0_5': 'descriptive summary, not an editing selection',
        'split_sha256': sha(ROOT / 'results/splits.json'),
        'checkpoint': json.loads((ROOT / 'checkpoint_manifest.json').read_text()),
        'script_sha256': sha(Path(__file__)),
        'inference_helper_sha256': sha(ROOT / 'probe_full_tokens.py'),
        'utils_sha256': sha(ROOT / 'utils.py'),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/token_probes')
    parser.add_argument('--batch-size', type=int, default=2, help='encoder inference batch')
    parser.add_argument('--token-batch-size', type=int, default=16, help='independent GPU ridge solves')
    parser.add_argument('--resume', action='store_true', help='skip atomically completed layers')
    args = parser.parse_args()
    if min(args.batch_size, args.token_batch_size) < 1:
        parser.error('Batch sizes must be positive')
    if not torch.cuda.is_available():
        parser.error('This sweep requires a GPU compute allocation')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    definition = protocol()
    if args.resume:
        previous = json.loads((args.output / 'protocol.json').read_text())
        if previous != definition:
            raise ValueError('Resume protocol/code mismatch; use a fresh output directory')
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / 'protocol.json').write_text(json.dumps(definition, indent=2))
    print(f'GPU: {torch.cuda.get_device_name()}; 147456 independent token probes; '
          f'{args.token_batch_size} solves per GPU batch', flush=True)
    data = load_data(ROOT)
    model = load_model(str(SCRATCH / 'checkpoints/vjepa2-vitl-fpc64-256'))
    del model.predictor
    torch.cuda.empty_cache()
    results = {}
    for task, ds in data.items():
        results[task] = []
        paths = [args.output / f'{task}_layer{layer:02d}.npz' for layer in range(1, 25)]
        if all(path.exists() for path in paths):
            results[task] = [summarize(path) for path in paths]
            print(f'{task}: all 24 layers already complete', flush=True)
            continue
        keep = np.flatnonzero(ds['split'] != 'C')
        videos = [ds['videos'][i] for i in keep]
        split, y, ids = ds['split'][keep], ds['y'][keep], ds['ids'][keep]
        reference = load_features(SCRATCH / f'pooled_features/{task}.npz', ds['ids'])
        reference = reference[keep[:3]].copy()
        print(f'{task}: {len(keep)} A/B/D videos; creating ~10 GiB host token buffer', flush=True)
        states = initial_tokens(model, videos, args.batch_size)
        for layer, (block, path) in enumerate(zip(model.encoder.layer, paths), start=1):
            layer_start = time.monotonic()
            advance_layer(block, states, 'cuda', args.batch_size)
            observed = states[:3].mean(axis=1)
            np.testing.assert_allclose(observed, reference[:, layer - 1], atol=2e-5, rtol=2e-5)
            if path.exists():
                results[task].append(summarize(path))
                print(f'{task} block {layer:02d}: replayed encoder, reused saved probes', flush=True)
                continue
            chunks = {}
            fitting_start = time.monotonic()
            for start in range(0, 2048, args.token_batch_size):
                stop = min(start + args.token_batch_size, 2048)
                fitted = fit_token_batch(states[:, start:stop], y, split, task)
                fitted.update(evaluate_tokens(y[split == 'D'], fitted['test_prediction'], task))
                for name, value in fitted.items():
                    chunks.setdefault(name, []).append(value)
                if start == 0 or stop % 128 == 0 or stop == 2048:
                    elapsed = time.monotonic() - fitting_start
                    print(f'{task} block {layer:02d}: {stop}/2048 tokens; '
                          f'{elapsed:.1f}s fitting; estimated remaining this layer '
                          f'{elapsed / stop * (2048 - stop) / 60:.1f}m', flush=True)
            arrays = {name: np.concatenate(values) for name, values in chunks.items()}
            temporary = path.with_suffix('.partial.npz')
            np.savez(temporary, **arrays, layer=layer, token_grid=np.array([8, 16, 16]),
                     train_ids=ids[split == 'A'], validation_ids=ids[split == 'B'],
                     test_ids=ids[split == 'D'], alpha_grid=np.array(ALPHAS),
                     metric_order=np.array(METRICS), seconds=time.monotonic() - layer_start,
                     pooled_reference_max_difference=float(abs(observed - reference[:, layer - 1]).max()))
            os.replace(temporary, path)
            summary = summarize(path)
            results[task].append(summary)
            (args.output / 'summary.json').write_text(json.dumps(results, indent=2))
            print(f'{task} block {layer:02d} saved: median B R²='
                  f'{summary["validation_r2_quantiles"][1]:.4f}; '
                  f'{summary["validation_fraction_r2_ge_0_5"]:.1%} of tokens have B R² >= .5; '
                  f'{summary["seconds"] / 60:.1f} minutes', flush=True)
            del chunks, arrays, fitted
        del states, reference
    (args.output / 'summary.json').write_text(json.dumps(results, indent=2))
    save_plot(results, args.output)
    print(f'Complete: all 147456 probes saved to {args.output}', flush=True)


if __name__ == '__main__':
    main()
