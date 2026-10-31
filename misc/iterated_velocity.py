"""Position-preserving iterative token velocity probes and predictor experiments.

Fit stages use only A/B. Frozen C evaluators are reused. This is an explicitly
exploratory follow-up on the existing D split, not a new untouched benchmark.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from misc.velocity_causal import CHECKPOINT, ROOT, SEED, configure, continue_predict, load_model, save_json
from misc.velocity_intervention import bootstrap_mean, effect_metrics, readout

ALPHAS = [1e-4, 1e-3, 1e-2, .1, 1., 10., 100., 1000., 10000.]
CAPS = [1, 2, 4, 8, 16, 32, 64, 128]
RANK_CAP = 256


def atom_npz(path, **arrays):
    temporary = path.with_suffix('.partial.npz')
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


def project(x, q):
    """x [token,sample,feature], q [token,feature,rank]."""
    return x - (x @ q) @ q.transpose(-1, -2)


def constrain_inverse(inverse, q):
    """Exact equality-constrained ridge inverse update, float64.

    H' = H - H Q (Q' H Q)^-1 Q' H. Zero columns in Q are supported.
    H has [token,alpha,feature,feature]; Q [token,feature,new_rank].
    """
    hq = inverse @ q[:, None]
    gram = q[:, None].transpose(-1, -2) @ hq
    invalid = q.square().sum(dim=-2) < .5
    gram = gram + torch.diag_embed(invalid.to(inverse.dtype))[:, None]
    correction = hq @ torch.linalg.solve(gram, hq.transpose(-1, -2))
    result = inverse - correction
    return (result + result.transpose(-1, -2))*.5


def ridge_inverse(xa, q, alphas=ALPHAS):
    """Build direct inverses on the retained subspace; refreshes roundoff drift."""
    xr = project(xa, q) if q.shape[-1] else xa
    gram = xr.transpose(-1, -2) @ xr
    values, vectors = torch.linalg.eigh(gram)
    values = values.clamp_min(0)
    aa = xa.new_tensor(alphas)
    inverse = (vectors[:, None] / (values[:, None, None, :] + aa[None, :, None, None])) @ vectors[:, None].transpose(-1, -2)
    if q.shape[-1]:
        # Project both sides to remove the artificial 1/alpha variance of the
        # directions already constrained away. This also handles padded zeros.
        qq = q[:, None]
        inverse = inverse - (inverse @ qq) @ qq.transpose(-1, -2)
        inverse = inverse - qq @ (qq.transpose(-1, -2) @ inverse)
    return (inverse+inverse.transpose(-1, -2))*.5


def fit_candidates(inverse, rhs, xb, yb, ymean, q):
    coef = inverse @ rhs[:, None]
    # Remove numerical drift from the constraint nullspace before validation.
    if q.shape[-1]:
        coef = coef - q[:, None] @ (q[:, None].transpose(-1, -2) @ coef)
    prediction = xb[:, None] @ coef + ymean
    mae = (prediction-yb[None, None]).abs().mean(dim=(2, 3))
    best = mae.argmin(dim=1)
    index = torch.arange(len(xb), device=xb.device)
    w = coef[index, best]
    pred = prediction[index, best]
    r2 = 1-(pred-yb).square().sum(1)/(yb-yb.mean(0)).square().sum(0)
    errors = (pred-yb).abs().mean(1)
    return w, best, r2, errors


def independent_basis(w, existing, max_rank, ranks):
    """Rows of the effective two-output probe, orthogonalized twice."""
    for _ in range(2):
        w = w-existing @ (existing.transpose(-1, -2) @ w)
    u, singular, _ = torch.linalg.svd(w, full_matrices=False)
    valid = singular > singular[:, :1].clamp_min(1e-30)*1e-8
    valid &= singular > 1e-12
    room = max_rank-ranks
    valid &= torch.arange(2, device=w.device)[None] < room[:, None]
    return u*valid[:, None], valid.sum(1), singular


@torch.inference_mode()
def direct_check(xa, xb, ya, yb, mean_y, q, w, alpha):
    """Independent dual solve of the constrained regression at selected alpha."""
    ar = project(xa, q)
    br = project(xb, q)
    gram = ar @ ar.transpose(-1, -2)
    eye = torch.eye(gram.shape[-1], dtype=gram.dtype, device=gram.device)
    dual = torch.linalg.solve(gram + alpha[:, None, None]*eye, ya-mean_y)
    expected = ar.transpose(-1, -2) @ dual
    p = xb @ w + mean_y
    ref = br @ expected+mean_y
    difference = (p-ref).abs().amax(dim=(1, 2))
    return difference, expected


def initialize(args):
    args.output.mkdir(parents=True, exist_ok=False)
    args.work.mkdir(parents=True, exist_ok=False)
    (args.work/'chunks').mkdir()
    definition = {
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'baseline': str(args.baseline), 'cache': str(args.cache), 'work': str(args.work),
        'layer': 19, 'prefix': 'frames 0-7 only', 'token_grid': [4, 16, 16],
        'fit': 'A only, feature centering and velocity component scaling from A',
        'validation': 'B selects each alpha by mean standardized component MAE',
        'position_constraint': 'reuse prior A/B-trained per-token position rows; all velocity probe directions orthogonal to them',
        'solver': 'float64 equality-constrained exact ridge with inverse rank updates, periodic direct refresh and independent dual-solve checks',
        'alphas': ALPHAS, 'max_removed_dimensions_per_token': RANK_CAP,
        'steering_eligibility': 'both B velocity-component R2 >= .5, both numerical output dimensions independent',
        'stopping': 'three successive probes with BOTH component R2 <= .05 AND BOTH component MAE >= .9 times training-mean baseline MAE on B; or rank exhaustion or dimension cap',
        'cap_semantics': 'cap is independent removed velocity dimensions, excludes protected position rank; fit a terminal probe at the cap',
        'steering_maximum_eligible_probes': CAPS,
        'velocity_changes_mps': [-.5, -.25, .25, .5],
        'edit': 'simultaneously shift all included effective velocity probes by the requested delta_v; minimum Euclidean norm; preserve position rows',
        'erasure': 'remove A-centered coordinates in the same eligible subspaces; also erase all discovered velocity directions at terminal state',
        'controls': 'steering: per-token norm-matched random directions orthogonal to position rows; erasure: random subspaces matched in rank and orthogonal to position rows',
        'evaluation': 'same B/D clips with speed in [.75,3.5], same frozen C position evaluators; no optimization on downstream outcomes',
        'joint_audit': 'exact full-token linear ridge on original and terminal-erased representations; A fit, B alpha, D evaluation; position-preserving and position-plus-velocity-erased diagnostics',
        'joint_audit_alpha_grid': [10.**k for k in range(-4, 9)],
        'history': 'D already inspected; this is a fixed exploratory follow-up',
        'seed': SEED,
    }
    save_json(args.output/'protocol.json', definition)


@torch.inference_mode()
def fit_chunk(features, y, split, position_coef, max_rank=RANK_CAP, alphas=ALPHAS, device='cuda'):
    """Fit independently per token, reading A/B features and labels ONLY."""
    ab = np.isin(split, ['A', 'B'])
    subset = split[ab]
    x = torch.tensor(np.ascontiguousarray(features[ab].transpose(1, 0, 2)), dtype=torch.float64, device=device)
    ya_raw = y[split=='A']; yb_raw = y[split=='B']
    sy = torch.tensor(ya_raw.std(0), dtype=torch.float64, device=device)
    ya = torch.tensor(ya_raw, dtype=torch.float64, device=device)/sy
    yb = torch.tensor(yb_raw, dtype=torch.float64, device=device)/sy
    ym = ya.mean(0)
    mean = x[:, subset=='A'].mean(1, keepdim=True)
    x -= mean
    xa, xb = x[:, subset=='A'].contiguous(), x[:, subset=='B'].contiguous()
    rhs = xa.transpose(-1, -2) @ (ya-ym)
    pc = torch.tensor(position_coef, dtype=torch.float64, device=device)
    u, sp, _ = torch.linalg.svd(pc.transpose(-1, -2), full_matrices=False)
    pvalid = sp > sp[:, :1]*1e-10
    qp = u*pvalid[:, None]
    t, _, d = x.shape
    qv = torch.zeros((t, d, max_rank), dtype=torch.float64, device=device)
    ranks = torch.zeros(t, dtype=torch.long, device=device)
    active = torch.ones(t, dtype=torch.bool, device=device)
    patience = torch.zeros(t, dtype=torch.long, device=device)
    reasons = np.full(t, '', dtype='<U24')
    baseline_mae = (yb-ym).abs().mean(0)
    history = []
    direct_max = 0.
    refreshed = 0
    q = qp
    inverse = ridge_inverse(xa, q, alphas)
    alpha_values = x.new_tensor(alphas)
    for iteration in range(max_rank+1):
        width = int(ranks.max())
        q = torch.cat((qp, qv[:, :, :width]), dim=-1)
        # Bound accumulated roundoff; update is exact in real arithmetic.
        if iteration and iteration % 16 == 0:
            inverse = ridge_inverse(xa, q, alphas)
            refreshed += 1
        w, best, r2, mae = fit_candidates(inverse, rhs, xb, yb, ym, q)
        near = (r2 <= .05).all(1) & (mae >= .9*baseline_mae).all(1)
        patience = torch.where(near, patience+1, 0)
        new_q, added, singular = independent_basis(w, q, max_rank, ranks)
        stopping = active & ((patience >= 3) | (ranks >= max_rank) | (added == 0))
        # Check every token when it terminates and all live tokens at refresh
        # boundaries; compare predictions to an independent sample-space solve.
        check = active & ((iteration % 16 == 0) | stopping)
        if check.any():
            difference, _ = direct_check(xa[check], xb[check], ya, yb, ym,
                                         q[check], w[check], alpha_values[best[check]])
            direct_max = max(direct_max, float(difference.max()))
            if difference.max() > 2e-4:
                inverse = ridge_inverse(xa, q, alphas)
                w, best, r2, mae = fit_candidates(inverse, rhs, xb, yb, ym, q)
                difference, _ = direct_check(xa[check], xb[check], ya, yb, ym,
                                             q[check], w[check], alpha_values[best[check]])
                if difference.max() > 2e-4:
                    raise AssertionError(f'Constrained/direct ridge disagreement {difference.max().item()}')
                new_q, added, singular = independent_basis(w, q, max_rank, ranks)
                refreshed += 1
        physical_w = w*sy[None, None]
        eligible = active & ~stopping & (r2 >= .5).all(1) & (added == 2)
        history.append({
            'coef': (physical_w.transpose(-1, -2)*active[:, None, None]).cpu().numpy(),
            'alpha': alpha_values[best].cpu().numpy(),
            'validation_r2': r2.cpu().numpy(), 'validation_mae': (mae*sy).cpu().numpy(),
            'active': active.cpu().numpy(), 'eligible': eligible.cpu().numpy(),
            'rank_before': ranks.cpu().numpy().copy(),
            'removed_rank': torch.where(active & ~stopping, added, 0).cpu().numpy(),
        })
        for j in torch.nonzero(stopping).flatten().cpu().tolist():
            reasons[j] = ('near_baseline' if patience[j] >= 3 else
                          'dimension_cap' if ranks[j] >= max_rank else 'rank_exhausted')
        active &= ~stopping
        if not active.any():
            break
        new_q *= active[:, None, None]
        for j in torch.nonzero(active).flatten().cpu().tolist():
            count = int(added[j]); start = int(ranks[j])
            qv[j, :, start:start+count] = new_q[j, :, :count]
        inverse = constrain_inverse(inverse, new_q)
        ranks += added*active
    q = torch.cat((qp, qv), dim=-1)
    gram = q.transpose(-1, -2) @ q
    expected_diag = q.square().sum(1)
    orth_error = float((gram-torch.diag_embed(expected_diag)).abs().max())
    if orth_error > 1e-7:
        raise AssertionError(f'Accumulated basis lost orthogonality: {orth_error}')
    result = {key: np.stack([h[key] for h in history]) for key in history[0]}
    result.update(mean=mean[:, 0].cpu().numpy(), position_basis=qp.cpu().numpy(),
                  velocity_basis=qv.cpu().numpy(), final_rank=ranks.cpu().numpy(), stop_reason=reasons,
                  y_scale=sy.cpu().numpy(), y_mean=(ym*sy).cpu().numpy(),
                  direct_check_max_difference=direct_max, orthogonality_max_error=orth_error,
                  inverse_refreshes=refreshed)
    return result


def fit(args):
    if not args.output.exists():
        initialize(args)
    else:
        previous = json.loads((args.output/'protocol.json').read_text())
        if previous['source_sha256'] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
            raise ValueError('Fit source changed; use fresh paths rather than mixing sequences')
    labels = np.load(args.baseline/'labels.npz')
    original = np.load(args.baseline/'edit_probes.npz')
    x = np.load(args.cache/'layer19.npy', mmap_mode='r')
    started = time.monotonic()
    for start in range(0, 1024, args.token_batch_size):
        stop = min(1024, start+args.token_batch_size)
        path = args.work/'chunks'/f'tokens_{start:04d}_{stop:04d}.npz'
        if path.exists():
            print(f'reuse {path.name}', flush=True)
            continue
        tick = time.monotonic()
        result = fit_chunk(np.asarray(x[:, start:stop]), labels['velocity'], labels['split'],
                           original['coef'][start:stop, 2:])
        atom_npz(path, **result, token_indices=np.arange(start, stop))
        elapsed = time.monotonic()-started
        eligible = result['eligible'].sum(0)
        print(f'tokens {start}:{stop}, {time.monotonic()-tick:.1f}s; '
              f'eligible={eligible.min()}..{eligible.max()}, ranks={result["final_rank"].min()}..{result["final_rank"].max()}, '
              f'elapsed {elapsed/60:.1f}m; approx remaining {elapsed/stop*(1024-stop)/60:.1f}m', flush=True)
        save_json(args.output/'fit_progress.json', {'tokens_complete': stop,
                  'elapsed_seconds': elapsed, 'latest_eligible_range': [int(eligible.min()), int(eligible.max())]})
    print('All token sequences complete', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['fit'])
    parser.add_argument('--baseline', type=Path, default=ROOT/'results/velocity_causal_v1')
    parser.add_argument('--cache', type=Path, default=Path('/scratch/users/hmolin/jepa-interp/velocity_causal_v1'))
    parser.add_argument('--output', type=Path, default=ROOT/'results/iterated_velocity_v1')
    parser.add_argument('--work', type=Path, default=Path('/scratch/users/hmolin/jepa-interp/iterated_velocity_v1'))
    parser.add_argument('--token-batch-size', type=int, default=16)
    args = parser.parse_args()
    configure()
    globals()[args.stage](args)


if __name__ == '__main__':
    main()
