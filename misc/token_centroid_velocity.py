"""Per-token empirical centroid splines with residual-preserving velocity edits.

Fit PCA/centroids on A independently at each token; select knots on B. Evaluate
raw and position-protected curve differences through the frozen predictor.
Run the full experiment on a compute allocation, not on a login node.
"""
from misc.paths import source_path
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.interpolate import PPoly
import torch

from misc.centroid_velocity import summary_item
from misc.iterated_velocity import atom_npz
from misc.spline_velocity import sha
from utils import fit_centroid_spline
from misc.velocity_causal import ROOT, CHECKPOINT, SEED, configure, save_json, load_model, continue_predict
from misc.velocity_intervention import readout, bootstrap_mean, effect_metrics

SIZES = [-.5, -.25, .25, .5]
KNOTS = [8, 12, 16]
VARIANTS = ['raw', 'protected']


def fit_token(xa, xb, sa, sb, dimensions=64, knot_options=KNOTS, device='cpu'):
    """Accept only A/B arrays; each token has its own PCA and knot selection."""
    a = torch.as_tensor(xa, dtype=torch.float64, device=device)
    mean = a.mean(0)
    centered = a - mean
    _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
    if dimensions > len(singular) or singular[dimensions-1] <= singular[0]*1e-10:
        raise ValueError('Insufficient rank for the requested PCA dimension')
    components = vh[:dimensions]
    za = (centered @ components.T).cpu().numpy()
    zb = ((torch.as_tensor(xb, dtype=torch.float64, device=device)-mean) @ components.T).cpu().numpy()
    candidates, curves = [], {}
    for knots in knot_options:
        curve, anchors, centroids, counts = fit_centroid_spline(za, sa, 'speed', knots, [.25, 4.])
        candidates.append({'knots': knots, 'B_PCA_reconstruction_mse': float(np.square(zb-curve(sb)).mean())})
        curves[knots] = (curve, anchors, centroids, counts)
    best = min(candidates, key=lambda row: row['B_PCA_reconstruction_mse'])
    curve, anchors, centroids, counts = curves[best['knots']]
    model = dict(components=components.cpu().numpy(), mean=mean.cpu().numpy(),
                 knots=curve.x, coefficients=curve.c, anchors=anchors, centroids=centroids, counts=counts)
    checks = dict(selected=best, candidates=candidates,
                  PCA_explained_variance=float(singular[:dimensions].square().sum()/singular.square().sum()),
                  component_orthogonality_error=float((components@components.T-torch.eye(dimensions, device=device)).abs().max()))
    return model, checks


def project_components(components, position_basis):
    """Project the curve itself; residual preservation is relative to that curve."""
    return components - (components @ position_basis) @ position_basis.T


def curve_value(model, speed, position_basis=None):
    speed = np.asarray(speed)
    if not np.isfinite(speed).all() or np.any((speed < .25) | (speed > 4.)):
        raise ValueError('Spline extrapolation is forbidden')
    components = model['components']
    if position_basis is not None:
        components = project_components(components, position_basis)
    return PPoly(model['coefficients'], model['knots'], extrapolate=False)(speed) @ components


def curve_difference(model, speed, size, position_basis=None):
    return curve_value(model, np.asarray(speed)+size, position_basis)-curve_value(model, speed, position_basis)


class TokenCurves:
    """Batched GPU evaluation, grouping tokens by their independently selected knots."""
    def __init__(self, models, position_basis, device='cuda'):
        self.tokens = len(models)
        self.channels = models[0]['components'].shape[1]
        self.device = device
        self.q = torch.as_tensor(position_basis, dtype=torch.float64, device=device)
        self.groups = []
        for count in sorted({len(m['knots']) for m in models}):
            indices = [i for i, m in enumerate(models) if len(m['knots']) == count]
            knots = models[indices[0]]['knots']
            for i in indices:
                np.testing.assert_array_equal(models[i]['knots'], knots)
            self.groups.append((torch.tensor(indices, device=device),
                                torch.tensor(knots, dtype=torch.float64, device=device),
                                torch.tensor(np.stack([models[i]['coefficients'] for i in indices]), device=device),
                                torch.tensor(np.stack([models[i]['components'] for i in indices]), device=device)))

    def values(self, speed):
        speed = torch.as_tensor(speed, dtype=torch.float64, device=self.device)
        if not bool(torch.isfinite(speed).all()) or bool(((speed < .25) | (speed > 4.)).any()):
            raise ValueError('Spline extrapolation is forbidden')
        result = torch.empty((len(speed), self.tokens, self.channels), dtype=torch.float64, device=self.device)
        for indices, knots, coefficients, components in self.groups:
            segment = (torch.searchsorted(knots, speed.contiguous(), right=True)-1).clamp(0, len(knots)-2)
            dx = (speed-knots[segment])[None, :, None]
            c = coefficients[:, :, segment, :]  # token, power, sample, PC
            values = ((c[:, 0]*dx+c[:, 1])*dx+c[:, 2])*dx+c[:, 3]
            result[:, indices] = (values @ components).permute(1, 0, 2)
        return result

    def difference(self, speed, size, protected=False):
        delta = self.values(np.asarray(speed)+size)-self.values(speed)
        if protected:
            delta -= torch.einsum('btr,tcr->btc', torch.einsum('btc,tcr->btr', delta, self.q), self.q)
        return delta


def random_edit(delta, directions, size):
    """Fixed seeded direction per token, matching each example/token edit norm."""
    return np.sign(size)*torch.linalg.vector_norm(delta, dim=-1, keepdim=True)*directions[None]


def input_paths(args):
    paths = {'labels': args.baseline/'labels.npz', 'position_basis': args.iterated_work/'position_basis.npy',
             'linear_zero': args.iterated/'conditions/zero.npz'}
    paths.update({f'readout_{k}': args.baseline/f'{k}_position_readout.npz' for k in ['predicted', 'teacher', 'context']})
    for size in SIZES:
        for cap in [1, 32]:
            paths[f'linear_k{cap}_{size}'] = args.iterated/'conditions'/f'velocity_k{cap:03d}_{size:+.2f}.npz'
    return paths


def make_protocol(args, labels, shape):
    sources = [Path(__file__), source_path('paths.py'), ROOT/'utils.py', ROOT / 'misc' / 'velocity_causal.py', ROOT / 'misc' / 'velocity_intervention.py',
               ROOT / 'misc' / 'centroid_velocity.py', ROOT / 'misc' / 'iterated_velocity.py', ROOT / 'misc' / 'spline_velocity.py']
    cache = args.cache/'layer19.npy'
    return {
        'method': 'Independent A-fitted PCA64 and empirical speed-centroid natural cubic spline at every prefix token',
        'shape': list(shape), 'dimensions': 64, 'knot_candidates': KNOTS,
        'selection': 'Each token independently minimizes B PCA reconstruction MSE at known speeds; no downstream selection',
        'fit_ids': labels['ids'][labels['split']=='A'].tolist(),
        'selection_ids': labels['ids'][labels['split']=='B'].tolist(),
        'edit': 'h_t + f_t(source_speed+delta)-f_t(source_speed); preserves the source residual exactly',
        'variants': {'raw': 'Original per-token PCA centroid curve',
                     'protected': 'Project the same selected curve off the frozen linear-method position rows; no refit'},
        'primary_variant': 'protected',
        'controls': 'Per-token norm-matched seeded random edits, with position protection matched to each variant',
        'random_seed': SEED+102, 'speed_changes': SIZES,
        'evaluation': 'Same B/D clips with source speeds [.75,3.5]; identical frozen checkpoint and C readouts',
        'linear_references': 'Frozen matched one-probe and all-eligible-probe conditions',
        'limitations': 'Speed-only oracle labels; direction not conditioned; one spline versus multiple linear probes; D previously inspected',
        'source_sha256': {p.name: sha(p) for p in sources},
        'inputs': {key: {'path': str(path.resolve()), 'sha256': sha(path)} for key, path in input_paths(args).items()},
        'cache': {'path': str(cache.resolve()), 'size': cache.stat().st_size, 'mtime_ns': cache.stat().st_mtime_ns},
        'checkpoint': str(CHECKPOINT),
    }


def verify_protocol(args):
    protocol = json.loads((args.output/'protocol.json').read_text())
    for name, digest in protocol['source_sha256'].items():
        if sha(source_path(name)) != digest:
            raise RuntimeError(f'Source changed since fit: {name}; use a new output directory')
    for key, path in input_paths(args).items():
        record = protocol['inputs'][key]
        if str(path.resolve()) != record['path'] or sha(path) != record['sha256']:
            raise RuntimeError(f'Frozen input changed: {key}')
    cache = args.cache/'layer19.npy'
    if protocol['cache'] != {'path': str(cache.resolve()), 'size': cache.stat().st_size, 'mtime_ns': cache.stat().st_mtime_ns}:
        raise RuntimeError('Activation cache changed')
    return protocol


def fit(args):
    labels = np.load(args.baseline/'labels.npz')
    x = np.load(args.cache/'layer19.npy', mmap_mode='r')
    if x.shape != (len(labels['speed']), 1024, 1024):
        raise ValueError(f'Unexpected activation shape: {x.shape}')
    a, b = [np.flatnonzero(labels['split'] == split) for split in ['A', 'B']]
    args.output.mkdir(parents=True, exist_ok=True)
    directory = args.output/'models'; directory.mkdir(exist_ok=True)
    protocol = make_protocol(args, labels, x.shape)
    path = args.output/'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise RuntimeError('Protocol changed; use a fresh output directory')
    save_json(path, protocol)
    code = args.output/'code'; code.mkdir(exist_ok=True)
    for name in protocol['source_sha256']:
        shutil.copy2(source_path(name), code/name)
    rows = []
    tick = time.monotonic()
    for start in range(0, x.shape[1], args.token_chunk):
        end = min(start+args.token_chunk, x.shape[1])
        needed = any(not (directory/f'token_{t:04d}.npz').exists() for t in range(start, end))
        # Slice tokens before advanced sample indexing, avoiding a full-cache copy.
        xa = np.array(x[:, start:end][a]) if needed else None
        xb = np.array(x[:, start:end][b]) if needed else None
        for token in range(start, end):
            model_path = directory/f'token_{token:04d}.npz'
            if model_path.exists():
                with np.load(model_path) as saved:
                    row = json.loads(str(saved['selection_json']))
            else:
                model, row = fit_token(xa[:, token-start], xb[:, token-start], labels['speed'][a], labels['speed'][b], device=args.fit_device)
                row['token'] = token
                atom_npz(model_path, **model, token=token, selection_json=json.dumps(row))
            rows.append(row)
        print(f'fit {end}/{x.shape[1]} tokens; {(time.monotonic()-tick)/60:.1f} min', flush=True)
    selection = {'tokens': rows, 'model_sha256': {p.name: sha(p) for p in sorted(directory.glob('token_*.npz'))}}
    save_json(args.output/'selection.json', selection)


def evaluate(args):
    verify_protocol(args)
    selection = json.loads((args.output/'selection.json').read_text())
    models = []
    for token in range(1024):
        path = args.output/'models'/f'token_{token:04d}.npz'
        if sha(path) != selection['model_sha256'][path.name]:
            raise RuntimeError(f'Model changed: {path}')
        with np.load(path) as z:
            models.append({key: z[key] for key in ['components', 'knots', 'coefficients']})
    q = np.load(args.iterated_work/'position_basis.npy')
    np.testing.assert_allclose(np.einsum('tcr,tcs->trs', q, q), np.broadcast_to(np.eye(2), (1024,2,2)), atol=1e-8)
    curves = TokenCurves(models, q)
    rng = np.random.default_rng(SEED+102)
    directions = torch.tensor(rng.normal(size=(1024,1024)), dtype=torch.float64, device='cuda')
    protected = directions-torch.einsum('tr,tcr->tc', torch.einsum('tc,tcr->tr', directions, curves.q), curves.q)
    directions /= torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
    protected /= torch.linalg.vector_norm(protected, dim=-1, keepdim=True)
    random_directions = {'raw': directions, 'protected': protected}
    labels = np.load(args.baseline/'labels.npz')
    keep = np.flatnonzero(np.isin(labels['split'], ['B','D']) & (labels['speed'] >= .75) & (labels['speed'] <= 3.5))
    old = np.load(args.iterated/'conditions/zero.npz')
    np.testing.assert_array_equal(old['indices'], keep)
    np.testing.assert_array_equal(old['ids'], labels['ids'][keep])
    cases = [{'name': 'zero', 'variant': 'raw', 'random': False, 'size': 0.}]
    cases += [{'name': f'{variant}{"_random" if random else ""}_{size:+.2f}', 'variant': variant, 'random': random, 'size': size}
              for size in SIZES for variant in VARIANTS for random in [False, True]]
    evaluation = {'cases': cases, 'indices': keep.tolist(), 'selection_sha256': sha(args.output/'selection.json')}
    selection_path = args.output/'evaluation_selection.json'
    if selection_path.exists() and json.loads(selection_path.read_text()) != evaluation:
        raise RuntimeError('Evaluation changed; use a fresh output directory')
    save_json(selection_path, evaluation)
    evaluators = {key: np.load(args.baseline/f'{key}_position_readout.npz') for key in ['predicted','teacher','context']}
    x = np.load(args.cache/'layer19.npy', mmap_mode='r')
    predictor = load_model(str(CHECKPOINT))
    directory = args.output/'conditions'; directory.mkdir(exist_ok=True)
    for number, case in enumerate(cases):
        path = directory/f'{case["name"]}.npz'
        if path.exists():
            print('reuse '+case['name'], flush=True)
            continue
        tick = time.monotonic()
        arrays = {key: np.empty((len(keep),4,2)) for key in evaluators}
        norms, source_norms = np.empty(len(keep)), np.empty(len(keep))
        constraint_error = random_norm_error = 0.
        for start in range(0, len(keep), args.batch_size):
            ii = keep[start:start+args.batch_size]; end = start+len(ii)
            h = torch.tensor(np.array(x[ii]), device='cuda')
            delta = curves.difference(labels['speed'][ii], case['size'], case['variant']=='protected')
            if case['size'] == 0. and bool((delta != 0).any()):
                raise AssertionError('Zero edit is not identity')
            if case['random']:
                control = random_edit(delta, random_directions[case['variant']], case['size'])
                random_norm_error = max(random_norm_error, float((torch.linalg.vector_norm(control, dim=-1)-torch.linalg.vector_norm(delta, dim=-1)).abs().max()))
                delta = control
            constraint_error = max(constraint_error, float(torch.einsum('btc,tcr->btr', delta, curves.q).abs().max()))
            edit = delta.to(h.dtype)
            norms[start:end] = torch.linalg.vector_norm(edit.flatten(1), dim=-1).cpu().numpy()
            source_norms[start:end] = torch.linalg.vector_norm(h.flatten(1), dim=-1).cpu().numpy()
            with torch.inference_mode():
                future, context = continue_predict(predictor, h+edit)
            future, context = future.cpu().numpy(), context.cpu().numpy()
            for key in evaluators:
                arrays[key][start:end] = readout(context if key=='context' else future, evaluators[key])
        if case['variant']=='protected' and constraint_error > 1e-7:
            raise AssertionError(f'Position constraint error: {constraint_error}')
        if random_norm_error > 1e-9:
            raise AssertionError(f'Random norm mismatch: {random_norm_error}')
        if case['name']=='zero':
            for key in arrays:
                np.testing.assert_allclose(arrays[key], old[key], atol=2e-3, rtol=1e-5)
        atom_npz(path, **arrays, indices=keep, ids=labels['ids'][keep], split=labels['split'][keep],
                 edit_norm=norms, source_norm=source_norms, position_basis_residual=constraint_error,
                 random_norm_error=random_norm_error, seconds=time.monotonic()-tick)
        print(f'{number+1}/{len(cases)} {case["name"]}: {time.monotonic()-tick:.1f}s', flush=True)
        save_json(args.output/'evaluation_progress.json', {'completed': number+1, 'total': len(cases), 'last': case['name']})
    summarize(args)


def summarize(args):
    verify_protocol(args)
    labels = np.load(args.baseline/'labels.npz')
    zero = np.load(args.output/'conditions/zero.npz'); keep = zero['indices']
    times = (np.arange(8,16).reshape(4,2).mean(1)-7)/24
    report = {'primary_variant': 'protected', 'splits': {}}
    for part in ['B', 'D']:
        mask = labels['split'][keep] == part
        result = {}
        for size in SIZES:
            v = labels['velocity'][keep[mask]]
            expected = (size*v/np.linalg.norm(v, axis=1, keepdims=True)) @ labels['world_to_pixel'][:2]
            loaded = {f'linear_k{cap:03d}': np.load(args.iterated/'conditions'/f'velocity_k{cap:03d}_{size:+.2f}.npz') for cap in [1,32]}
            loaded.update({f'{variant}{suffix}': np.load(args.output/'conditions'/f'{variant}{suffix}_{size:+.2f}.npz')
                           for variant in VARIANTS for suffix in ['', '_random']})
            for kind, z in loaded.items():
                np.testing.assert_array_equal(z['indices'], keep)
                np.testing.assert_array_equal(z['ids'], labels['ids'][keep])
                item = summary_item(z, zero, mask, expected, times)
                item['videos'] = int(mask.sum())
                item['relative_edit_norm'] = bootstrap_mean((z['edit_norm']/z['source_norm'])[mask])
                item['position_basis_residual_max'] = float(z['position_basis_residual'])
                if kind in VARIANTS:
                    for reference in [kind+'_random', 'linear_k001', 'linear_k032']:
                        item['paired_vs_'+reference] = effect_metrics((z['predicted']-loaded[reference]['predicted'])[mask], expected, times)
                result[f'{kind}_{size:+.2f}'] = item
        report['splits'][part] = result
    save_json(args.output/'effects.json', report)
    write_report(args, report)


def write_report(args, report):
    rows = ['# Per-token offset-preserving centroid spline steering', '',
            'Each prefix token has its own A-fitted PCA64 and speed-centroid spline. B independently selects 8/12/16 knots per token. '
            'Edits add the source-to-target curve difference, preserving the original residual. No C/D examples fit or select curves.', '',
            'The primary protected variant projects the selected curve off the same position-readout directions as the linear method. '
            'It preserves the residual relative to this projected curve. The raw curve is a fixed secondary comparison. '
            'Controls match per-token edit norms and position protection; neither downstream performance nor D selects a variant.', '',
            '## D-set results', '', '| Δspeed (m/s) | Method | Velocity gain % [95% CI] | Context drift (px) | Relative edit norm |',
            '|---:|---|---:|---:|---:|']
    for size in SIZES:
        for kind in ['raw','raw_random','protected','protected_random','linear_k001','linear_k032']:
            item = report['splits']['D'][f'{kind}_{size:+.2f}']
            gain = item['predicted']['velocity_slope_gain']; lo, hi = gain['ci95']
            rows.append(f'| {size:+.2f} | {kind} | {100*gain["mean"]:.2f} [{100*lo:.2f}, {100*hi:.2f}] | '
                        f'{item["context_last_shift_pixels"]["mean"]:.3f} | {item["relative_edit_norm"]["mean"]:.5f} |')
    rows += ['', '## Paired primary comparisons', '',
             '| Δspeed (m/s) | Protected spline minus reference | Gain difference (percentage points) [95% CI] |',
             '|---:|---|---:|']
    for size in SIZES:
        item = report['splits']['D'][f'protected_{size:+.2f}']
        for reference in ['protected_random','linear_k001','linear_k032']:
            gain = item['paired_vs_'+reference]['velocity_slope_gain']; lo, hi = gain['ci95']
            rows.append(f'| {size:+.2f} | {reference} | {100*gain["mean"]:.2f} [{100*lo:.2f}, {100*hi:.2f}] |')
    rows += ['', '## Interpretation limits', '',
             'This matches token specificity, clips, targets, frozen predictor/readouts, and (for the primary variant) construction-position constraints. '
             'It does not isolate function class: one speed-only centroid curve differs from iterated direction-aware velocity probes, and edit norms are not matched between methods. '
             'The one-probe linear reference is also reported. Source speeds are oracle labels.', '',
             'Residual preservation does not guarantee preservation of every scene property or membership in the natural activation distribution. '
             'Position protection applies to construction readouts; independent context drift and perpendicular response are recorded. '
             'The frozen position readout was trained on unedited predictor outputs; teacher-token transfer remains weak. No edited videos are generated.', '',
             'D was previously inspected: this is exploratory. Intervals use 1,000 paired clip bootstraps conditional on frozen fits; '
             'they omit training uncertainty and multiple-comparison correction. Random controls use one seeded direction per token.', '',
             'The protocol, per-token selection records, model/source hashes, predictions, B/D metrics and paired comparisons are saved alongside this report.']
    (args.output/'REPORT.md').write_text('\n'.join(rows)+'\n')
    print(args.output/'REPORT.md', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['fit','evaluate','summarize'])
    parser.add_argument('--output', type=Path, default=ROOT/'results/token_centroid_velocity_v1')
    parser.add_argument('--cache', type=Path, default=Path('/scratch/users/hmolin/jepa-interp/velocity_causal_v1'))
    parser.add_argument('--baseline', type=Path, default=ROOT/'results/velocity_causal_v1')
    parser.add_argument('--iterated', type=Path, default=ROOT/'results/iterated_velocity_v1')
    parser.add_argument('--iterated-work', type=Path, default=Path('/scratch/users/hmolin/jepa-interp/iterated_velocity_v1'))
    parser.add_argument('--fit-device', choices=['cpu','cuda'], default='cuda')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--token-chunk', type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1 or args.token_chunk < 1:
        parser.error('Batch and token chunk sizes must be positive')
    configure()
    globals()[args.stage](args)


if __name__ == '__main__':
    main()
