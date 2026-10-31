"""Layer sweep of ridge probes on all tokens, without spatial/temporal pooling.

Each probe sees 2048 * 1024 features. Exact dual ridge keeps the solve at
768 training examples; coefficients retain separate weights for every token.
Run on one GPU with 32 GB host RAM. Only one task/layer is retained in RAM.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from utils import bootstrap_interval, load_data, load_features, load_model, load_video, metrics


ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path('/scratch/users/hmolin/jepa-interp')
# Retain the original grid and extend upward for the much larger feature space.
ALPHAS = [10. ** exponent for exponent in range(-4, 9)]
TASKS = ('direction', 'speed', 'acceleration')


@torch.inference_mode()
def fit_full_probe(features, y, split, task, alphas=ALPHAS, device='cuda', chunk_size=8192):
    """Exact centered ridge in sample space. No approximation or feature scaling.

    D labels are never read here. Test predictions are returned for later scoring.
    Coefficients act on raw flattened tokens: prediction = X @ coef.T + intercept.
    """
    x = features.reshape(len(features), -1)
    train = np.flatnonzero(split == 'A')
    val = np.flatnonzero(split == 'B')
    if len(train) < 2 or len(val) == 0:
        raise ValueError('Need at least two A examples and one B example')
    train_index = torch.as_tensor(train, device=device)
    kernel = torch.zeros((len(x), len(train)), dtype=torch.float64, device=device)
    means = np.empty(x.shape[1], dtype=np.float64)
    # Center before multiplying; float64 also avoids cancellation in the solve.
    for start in range(0, x.shape[1], chunk_size):
        stop = min(start + chunk_size, x.shape[1])
        block = torch.as_tensor(np.ascontiguousarray(x[:, start:stop]), device=device,
                                dtype=torch.float64)
        mean = block[train_index].mean(0)
        means[start:stop] = mean.cpu().numpy()
        block = block - mean
        kernel.addmm_(block, block[train_index].T)
    kernel = kernel.cpu().numpy()
    gram = kernel[train]
    eigenvalues, eigenvectors = np.linalg.eigh((gram + gram.T) / 2)
    if eigenvalues.min() < -1e-8 * max(1., eigenvalues.max()):
        raise ValueError('Training kernel is not numerically positive semidefinite')
    eigenvalues = np.maximum(eigenvalues, 0.)
    y_mean = y[train].mean(0)
    rhs = eigenvectors.T @ (y[train] - y_mean)
    validation = []
    for alpha in alphas:
        dual = eigenvectors @ (rhs / (eigenvalues[:, None] + alpha))
        prediction = kernel[val] @ dual + y_mean
        validation.append({'alpha': float(alpha), **metrics(y[val], prediction, task)})
    best = min(range(len(alphas)), key=lambda i: validation[i]['mae'])
    alpha = alphas[best]
    dual = eigenvectors @ (rhs / (eigenvalues[:, None] + alpha))
    predictions = kernel @ dual + y_mean

    # Recover explicit token-space weights for later interventions.
    dual_tensor = torch.as_tensor(dual, device=device)
    coef = np.empty((y.shape[1], x.shape[1]), dtype=np.float64)
    for start in range(0, x.shape[1], chunk_size):
        stop = min(start + chunk_size, x.shape[1])
        block = torch.as_tensor(np.ascontiguousarray(x[train, start:stop]),
                                device=device, dtype=torch.float64)
        block = block - torch.as_tensor(means[start:stop], device=device)
        coef[:, start:stop] = (block.T @ dual_tensor).T.cpu().numpy()
    intercept = y_mean - means @ coef.T
    # Check that the saved, raw-token readout agrees with the dual solution.
    check = np.unique(np.r_[train[:2], val[:2], np.flatnonzero(split == 'D')[:2]])
    direct = x[check].astype(np.float64) @ coef.T + intercept
    np.testing.assert_allclose(direct, predictions[check], atol=1e-6, rtol=1e-5)
    record = {
        'alpha': float(alpha),
        'alpha_at_grid_edge': best in (0, len(alphas) - 1),
        'validation': {key: validation[best][key] for key in ('r2', 'mae')},
        'validation_grid': validation,
        'train': metrics(y[train], predictions[train], task),
        'primal_dual_max_difference': float(abs(direct - predictions[check]).max()),
    }
    return record, coef, intercept, predictions


def normalized_pixels(paths, device):
    frames = np.stack([load_video(path) for path in paths])
    pixels = torch.from_numpy(frames).permute(0, 1, 4, 2, 3).to(device).float() / 255
    mean = pixels.new_tensor([.485, .456, .406])[None, None, :, None, None]
    std = pixels.new_tensor([.229, .224, .225])[None, None, :, None, None]
    return (pixels - mean) / std


@torch.inference_mode()
def initial_tokens(model, videos, batch_size):
    device = next(model.parameters()).device
    states = np.empty((len(videos), 2048, 1024), dtype=np.float32)
    for start in range(0, len(videos), batch_size):
        paths = videos[start:start + batch_size]
        embedded = model.encoder.embeddings(normalized_pixels(paths, device))
        assert embedded.shape[1:] == (2048, 1024)
        states[start:start + len(paths)] = embedded.cpu().numpy()
        if start % 200 == 0:
            print(f'  Patch embeddings: {start + len(paths)}/{len(videos)}', flush=True)
    return states


@torch.inference_mode()
def advance_layer(block, states, device, batch_size):
    """Advance every clip one encoder block, overwriting the old host buffer."""
    for start in range(0, len(states), batch_size):
        batch = torch.from_numpy(states[start:start + batch_size]).to(device)
        states[start:start + len(batch)] = block(batch)[0].cpu().numpy()


def select_layers(results):
    common = int(np.argmax(np.mean([[r['validation']['r2'] for r in results[t]['layers']]
                                   for t in TASKS], axis=0)))
    use_common = all(results[t]['layers'][common]['validation']['r2'] >= .5 for t in TASKS)
    return {t: common + 1 if use_common else
            int(np.argmin([r['validation']['mae'] for r in results[t]['layers']])) + 1
            for t in TASKS}


def save_plot(results, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    pooled = json.loads((ROOT / 'results/probe_results.json').read_text())
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout='constrained')
    units = {'direction': 'degrees', 'speed': 'm/s', 'acceleration': 'm/s²'}
    for col, task in enumerate(TASKS):
        for row, metric in enumerate(('r2', 'mae')):
            ax = axes[row, col]
            layers = results[task]['layers']
            bounds = np.array([r['test_ci95'][metric] for r in layers])
            ax.plot(range(1, 25), [r['test'][metric] for r in layers], label='Full tokens: test')
            ax.fill_between(range(1, 25), bounds[:, 0], bounds[:, 1], alpha=.15)
            ax.plot(range(1, 25), [r['validation'][metric] for r in layers], '--',
                    label='Full tokens: validation')
            ax.plot(range(1, 25), [r['test'][metric] for r in pooled[task]['layers']],
                    label='Pooled: historical test', color='gray')
            ax.axvline(results[task]['selected_layer'], color='black', alpha=.25)
            ax.set(title=task, xlabel='Block',
                   ylabel='R²' if metric == 'r2' else f'MAE ({units[task]})')
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(output / 'layerwise.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/full_token_probes')
    parser.add_argument('--batch-size', type=int, default=2)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    if not torch.cuda.is_available():
        parser.error('This sweep requires a GPU compute allocation')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    data = load_data(ROOT)
    protocol = {
        'representation': 'flattened raw block residuals; no spatial or temporal pooling',
        'token_shape': [8, 16, 16, 1024], 'feature_count': 2048 * 1024,
        'token_order': 'time, height, width, channel (C-order)',
        'layers': list(range(1, 25)), 'alphas': ALPHAS,
        'fit': 'A', 'validation': 'B', 'reserved': 'C', 'test': 'D',
        'centering': 'training mean only; intercept fitted; no feature scaling',
        'solver': 'exact dual ridge, float64 centered Gram accumulation and eigensolve',
        'inference': 'float32, evaluation mode, TF32 disabled, raw outputs before final LayerNorm',
        'preprocessing': 'native 16 RGB frames at 256x256, ImageNet normalization',
        'alpha_selection': 'minimum B MAE, grid extended above historical 1e4 to 1e8',
        'layer_selection': 'maximize mean B R2; common layer if all B R2 >= .5, otherwise task B MAE minima',
        'splits_sha256': hashlib.sha256((ROOT / 'results/splits.json').read_bytes()).hexdigest(),
        'checkpoint': json.loads((ROOT / 'checkpoint_manifest.json').read_text()),
        'batch_size': args.batch_size, 'bootstrap_replicates': 500,
    }
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    model = load_model(str(SCRATCH / 'checkpoints/vjepa2-vitl-fpc64-256'))
    # The predictor is unused; release it to leave more GPU memory for probe fits.
    del model.predictor
    torch.cuda.empty_cache()
    print(f'GPU: {torch.cuda.get_device_name()}; full token sweep, 72 probes', flush=True)
    results = {}
    for task, ds in data.items():
        keep = np.flatnonzero(ds['split'] != 'C')
        videos = [ds['videos'][i] for i in keep]
        split, y, ids = ds['split'][keep], ds['y'][keep], ds['ids'][keep]
        # Small reference subset checks every block against the existing hook extraction.
        reference = load_features(SCRATCH / f'pooled_features/{task}.npz', ds['ids'])
        reference = reference[keep[:3]].copy()
        print(f'{task}: {len(keep)} A/B/D clips; allocating '
              f'{len(keep) * 2048 * 1024 * 4 / 2**30:.2f} GiB token buffer', flush=True)
        states = initial_tokens(model, videos, args.batch_size)
        results[task] = {'layers': []}
        for layer, block in enumerate(model.encoder.layer, start=1):
            layer_started = time.monotonic()
            advance_layer(block, states, 'cuda', args.batch_size)
            observed = states[:3].mean(axis=1)
            np.testing.assert_allclose(observed, reference[:, layer - 1], atol=2e-5, rtol=2e-5)
            record, coef, intercept, predictions = fit_full_probe(states, y, split, task)
            record['layer'] = layer
            record['pooled_reference_max_difference'] = float(abs(observed - reference[:, layer - 1]).max())
            record['seconds'] = time.monotonic() - layer_started
            # Test predictions are saved now, but all test scoring waits for layer selection.
            np.savez(args.output / f'probe_{task}_layer{layer:02d}.npz',
                     coef=coef.reshape(y.shape[1], 2048, 1024), intercept=intercept,
                     alpha=record['alpha'], layer=layer, token_grid=np.array([8, 16, 16]),
                     train_ids=ids[split == 'A'], validation_ids=ids[split == 'B'],
                     test_ids=ids[split == 'D'], test_prediction=predictions[split == 'D'])
            results[task]['layers'].append(record)
            (args.output / 'validation.json').write_text(json.dumps(results, indent=2))
            print(f'{task} block {layer:02d}: B R²={record["validation"]["r2"]:.4f}, '
                  f'MAE={record["validation"]["mae"]:.4f}, alpha={record["alpha"]:g}, '
                  f'{record["seconds"]:.1f}s'
                  + (' (alpha at grid edge)' if record['alpha_at_grid_edge'] else ''), flush=True)
            del coef, predictions
        del states, reference

    selected = select_layers(results)
    (args.output / 'selection.json').write_text(json.dumps(selected, indent=2))
    for task, ds in data.items():
        results[task]['selected_layer'] = selected[task]
        a, d = ds['split'] == 'A', ds['split'] == 'D'
        baseline = np.repeat(ds['y'][a].mean(0, keepdims=True), int(d.sum()), axis=0)
        results[task]['constant_test'] = metrics(ds['y'][d], baseline, task)
        for record in results[task]['layers']:
            layer = record['layer']
            with np.load(args.output / f'probe_{task}_layer{layer:02d}.npz') as saved:
                np.testing.assert_array_equal(saved['test_ids'], ds['ids'][d])
                prediction = saved['test_prediction']
            record['test'] = metrics(ds['y'][d], prediction, task)
            record['test_ci95'] = bootstrap_interval(ds['y'][d], prediction, task)
        print(f'{task}: selected block {selected[task]}, '
              f'test {results[task]["layers"][selected[task] - 1]["test"]}', flush=True)
    (args.output / 'scores.json').write_text(json.dumps(results, indent=2))
    save_plot(results, args.output)
    print(f'Finished in {(time.monotonic() - started) / 60:.1f} minutes. '
          f'Outputs: {args.output}', flush=True)


if __name__ == '__main__':
    main()
