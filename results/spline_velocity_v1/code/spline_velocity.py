"""Conditional token-level speed splines and frozen predictor interventions.

The input basis is shared across tokens; every token/channel has its own output
coefficients. No pooled activation is broadcast to tokens. Model selection uses
only A/B activation reconstruction, never downstream D intervention outcomes.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
from scipy.interpolate import CubicSpline
import torch

from iterated_velocity import atom_npz
from velocity_causal import ROOT, SEED, CHECKPOINT, configure, load_model, continue_predict, save_json
from velocity_intervention import readout, effect_metrics, bootstrap_mean

KNOTS = [8, 12, 16]
ORDERS = [1, 2]
ALPHAS = [1e-3, 1e-2, .1, 1., 10., 100., 1000., 10000.]
SIZES = [-.5, -.25, .25, .5]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def covariates(labels, indices):
    v = labels['velocity'][indices]
    return np.column_stack((labels['speed'][indices], np.arctan2(v[:, 1], v[:, 0]),
                            (labels['positions'][indices, 7]-128.)/128.))


def design(cov, knots, order):
    """Natural cubic speed basis x Fourier direction x quadratic boundary position."""
    cov = np.asarray(cov, dtype=np.float64)
    speed, theta, x, y = cov.T
    if np.any((speed < .25-1e-10) | (speed > 4.+1e-10)):
        raise ValueError('Speed outside fitted domain; extrapolation is forbidden')
    basis = CubicSpline(np.linspace(.25, 4., knots), np.eye(knots),
                        bc_type='natural', extrapolate=False)(np.clip(speed, .25, 4.))
    direction = np.column_stack([np.ones(len(cov)),
                                *[f(k*theta) for k in range(1, order+1) for f in (np.sin, np.cos)]])
    position = np.column_stack((np.ones(len(cov)), x, y, x*x, x*y, y*y))
    return np.einsum('ni,nj,nk->nijk', basis, direction, position).reshape(len(cov), -1)


def standardize(a, b):
    mean, scale = a.mean(0), a.std(0)
    scale[scale < 1e-10] = 1.
    return (a-mean)/scale, (b-mean)/scale, mean, scale


def protected(y, q):
    """Project [sample,token,channel] away from [token,channel,2] rows."""
    return y - torch.einsum('ntr,tcr->ntc', torch.einsum('ntc,tcr->ntr', y, q), q)


def ridge_map(a, alpha):
    return torch.linalg.solve(a.T@a+alpha*torch.eye(a.shape[1], dtype=a.dtype, device=a.device), a.T)


def gram_mse(weights, gram, train_count):
    """Exact multi-output B MSE, from A-centered output Gram matrix (per output supplied externally)."""
    aa = gram[:train_count, :train_count]
    ba = gram[train_count:, :train_count]
    bb = gram[train_count:, train_count:]
    return (torch.trace(bb)-2*(weights*ba).sum()+((weights@aa)*weights).sum())/len(weights)


def select_model(train_cov, validation_cov, gram):
    """Only explicit A/B inputs accepted; no access to C/D data."""
    n = len(train_cov)
    candidates, best = [], None
    for knots, order in itertools.product(KNOTS, ORDERS):
        a, b, mean, scale = standardize(design(train_cov, knots, order), design(validation_cov, knots, order))
        at, bt = [torch.tensor(z, dtype=torch.float64, device=gram.device) for z in (a, b)]
        eigenvalues, u = torch.linalg.eigh(at.T@at)
        eigenvalues = eigenvalues.clamp_min(0)
        left, right = bt@u, u.T@at.T
        for alpha in ALPHAS:
            weights = (left/(eigenvalues+alpha))@right
            error = float(gram_mse(weights, gram, n))
            item = dict(knots=knots, direction_order=order, alpha=alpha,
                        features=a.shape[1], validation_mse=error)
            candidates.append(item)
            if best is None or error < best['validation_mse']:
                best = item
        print(f'spline selection knots={knots} direction_order={order} complete', flush=True)
    return best, candidates


def fit(args):
    args.output.mkdir(parents=True, exist_ok=True); args.work.mkdir(parents=True, exist_ok=True)
    labels = np.load(args.baseline/'labels.npz')
    a = np.flatnonzero(labels['split']=='A'); b = np.flatnonzero(labels['split']=='B')
    indices = np.r_[a, b]
    qp = np.load(args.iterated_work/'position_basis.npy', mmap_mode='r')
    protocol = {
        'method': 'conditional natural cubic speed spline x periodic Fourier direction x quadratic boundary position; separate output per token/channel',
        'activation': 'prefix-only layer19, 1024 tokens x 1024 channels; all outputs, no pooling or output PCA',
        'speed_knots': KNOTS, 'direction_harmonic_orders': ORDERS, 'ridge_alphas': ALPHAS,
        'position_features': '1,x,y,x^2,xy,y^2; measured frame7 position normalized by (pixels-128)/128',
        'fit': 'A-only centered/scaled design, A-centered activation outputs projected off existing protected position rows',
        'selection': 'B mean squared reconstruction error across all protected token/channel outputs; one global choice',
        'edit': 'f(s+delta,theta,p7)-f(s,theta,p7), preserving original example residual; no extrapolation or downstream gain tuning',
        'source_labels': 'oracle metadata speed/direction and observed-frame7 measured position',
        'protected_position_basis_sha256': sha(args.iterated_work/'position_basis.npy'),
        'source_sha256': sha(__file__), 'labels_sha256': sha(args.baseline/'labels.npz'),
        'fit_ids': labels['ids'][a].tolist(), 'selection_ids': labels['ids'][b].tolist(),
        'evaluation': 'same speed-restricted B/D cases, speed in [.75,3.5]; D historically inspected, exploratory',
        'delta_speed_mps': SIZES, 'linear_baseline': 'all B-eligible iterated probes, saturated cap32, frozen prior results',
        'random_control': 'same seeded random directions as iterated run, position-null, matched per-token spline edit norms',
        'frozen_readout_sha256': {k:sha(args.baseline/f'{k}_position_readout.npz') for k in ('predicted','teacher','context')},
        'work': str(args.work), 'cache': str(args.cache),
    }
    path = args.output/'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise RuntimeError('Protocol changed; use a fresh output/work directory')
    save_json(path, protocol)
    x = np.load(args.cache/'layer19.npy', mmap_mode='r')
    started = time.monotonic()
    gram_path = args.work/'selection_gram.npy'
    if not gram_path.exists():
        gram = torch.zeros((len(indices),len(indices)), dtype=torch.float64, device='cuda')
        for start in range(0,1024,16):
            y = torch.tensor(np.array(x[indices,start:start+16]), dtype=torch.float64, device='cuda')
            y -= y[:len(a)].mean(0,keepdim=True)
            q = torch.tensor(np.array(qp[start:start+16]), dtype=torch.float64, device='cuda')
            y = protected(y,q).flatten(1)
            gram += y@y.T/(1024*1024)
            if (start+16)%128 == 0:
                print(f'A/B protected-output Gram {start+16}/1024; {(time.monotonic()-started)/60:.1f}m', flush=True)
        np.save(gram_path,gram.cpu().numpy())
    gram = torch.tensor(np.load(gram_path),dtype=torch.float64,device='cuda')
    best, candidates = select_model(covariates(labels,a),covariates(labels,b),gram)
    selection = {'selected':best, 'candidates':candidates,
                 'training_mean_validation_mse':float(gram[len(a):,len(a):].diagonal().mean()),
                 'seconds_to_selection':time.monotonic()-started}
    selection['validation_fraction_explained_vs_A_mean'] = 1-best['validation_mse']/selection['training_mean_validation_mse']
    save_json(args.output/'selection.json',selection)
    print('SELECTED '+json.dumps(selection['selected']),flush=True)
    acov, bcov = covariates(labels,a),covariates(labels,b)
    aa, bb, mean, scale = standardize(design(acov,best['knots'],best['direction_order']),design(bcov,best['knots'],best['direction_order']))
    at = torch.tensor(aa,dtype=torch.float64,device='cuda')
    mapping = ridge_map(at,best['alpha'])
    coeff = np.lib.format.open_memmap(args.work/'coefficients.npy',mode='w+',dtype=np.float32,shape=(aa.shape[1],1024,1024))
    actual_error, baseline_error, max_residual = 0., 0., 0.
    for start in range(0,1024,16):
        y = torch.tensor(np.array(x[indices,start:start+16]),dtype=torch.float64,device='cuda')
        y -= y[:len(a)].mean(0,keepdim=True)
        q = torch.tensor(np.array(qp[start:start+16]),dtype=torch.float64,device='cuda')
        y = protected(y,q)
        c = (mapping@y[:len(a)].flatten(1)).reshape(aa.shape[1],16,1024)
        max_residual = max(max_residual,float(torch.einsum('dtc,tcr->dtr',c,q).abs().max()))
        # Verify exact Gram-based model selection against direct predictions.
        pred = torch.tensor(bb,dtype=torch.float64,device='cuda')@c.flatten(1)
        actual_error += float((pred-y[len(a):].flatten(1)).square().sum())/(len(b)*1024*1024)
        baseline_error += float(y[len(a):].square().sum())/(len(b)*1024*1024)
        coeff[:,start:start+16] = c.cpu().numpy().astype(np.float32)
    coeff.flush()
    np.testing.assert_allclose(actual_error,best['validation_mse'],rtol=1e-7,atol=1e-10)
    if max_residual > 1e-7:
        raise AssertionError(f'Protected-position coefficient residual {max_residual}')
    atom_npz(args.output/'model.npz',mean=mean,scale=scale,knots=best['knots'],direction_order=best['direction_order'],
             alpha=best['alpha'],training_ids=labels['ids'][a],validation_ids=labels['ids'][b])
    save_json(args.output/'fit_checks.json',{'direct_validation_mse':actual_error,'gram_validation_mse':best['validation_mse'],
              'position_constraint_max_residual':max_residual,'seconds':time.monotonic()-started,
              'coefficient_sha256':sha(args.work/'coefficients.npy')})
    print('fit complete',flush=True)


def delta_design(cov, size, model):
    target = cov.copy(); target[:,0] += size
    return (design(target,int(model['knots']),int(model['direction_order']))-
            design(cov,int(model['knots']),int(model['direction_order'])))/model['scale']


def evaluate(args):
    protocol = json.loads((args.output/'protocol.json').read_text())
    if protocol['source_sha256'] != sha(__file__):
        raise RuntimeError('Experiment source changed since fit')
    for k,digest in protocol['frozen_readout_sha256'].items():
        assert sha(args.baseline/f'{k}_position_readout.npz') == digest
    labels = np.load(args.baseline/'labels.npz')
    keep = np.flatnonzero(np.isin(labels['split'],['B','D']) & (labels['speed']>=.75) & (labels['speed']<=3.5))
    zero = np.load(args.iterated/'conditions/zero.npz')
    np.testing.assert_array_equal(keep,zero['indices'])
    saved = np.load(args.output/'model.npz')
    cov = covariates(labels,keep)
    np.testing.assert_array_equal(delta_design(cov,0.,saved),np.zeros((len(keep),len(saved['scale']))))
    conditions = [{'name':'zero','kind':'zero','size':0.}]
    conditions += [{'name':f'{kind}_{size:+.2f}','kind':kind,'size':size} for size in SIZES for kind in ('spline','random')]
    save_json(args.output/'evaluation_selection.json',{'conditions':conditions,'indices':keep.tolist(),
              'ids':labels['ids'][keep].tolist(),'split':labels['split'][keep].tolist(),
              'selection_sha256':sha(args.output/'selection.json'),
              'linear_baseline_files':{f'{kind}_{size:+.2f}':sha(args.iterated/'conditions'/f'{kind}_k032_{size:+.2f}.npz')
                  for kind in ('velocity','random') for size in SIZES}})
    directory = args.output/'conditions'; directory.mkdir(exist_ok=True)
    x = np.load(args.cache/'layer19.npy',mmap_mode='r')
    c = torch.tensor(np.load(args.work/'coefficients.npy'),device='cuda')
    qp = torch.tensor(np.load(args.iterated_work/'position_basis.npy'),dtype=torch.float32,device='cuda')
    rng = np.random.default_rng(SEED+102)
    random = torch.tensor(rng.normal(size=(1024,1024)),dtype=torch.float32,device='cuda')
    random = protected(random[None],qp)[0]
    random /= torch.linalg.vector_norm(random,dim=-1,keepdim=True)
    evaluators = {k:np.load(args.baseline/f'{k}_position_readout.npz') for k in ('predicted','teacher','context')}
    model = load_model(str(CHECKPOINT))
    for num,condition in enumerate(conditions):
        name,kind,size = condition['name'],condition['kind'],condition['size']
        dest = directory/f'{name}.npz'
        if dest.exists():
            print('reuse '+name,flush=True); continue
        tick = time.monotonic()
        basis = torch.tensor(delta_design(cov,size,saved),dtype=torch.float32,device='cuda')
        arrays = {k:np.empty((len(keep),4,2)) for k in evaluators}
        norms,source = np.empty(len(keep)),np.empty(len(keep))
        residual = 0.
        for start in range(0,len(keep),args.batch_size):
            ii=keep[start:start+args.batch_size]; end=start+len(ii)
            h = torch.tensor(np.array(x[ii]),device='cuda')
            edit = torch.einsum('bd,dtc->btc',basis[start:end],c)
            # Re-project to remove float32 storage/accumulation residuals.
            edit = protected(edit,qp)
            if kind=='random':
                edit = np.sign(size)*torch.linalg.vector_norm(edit,dim=-1,keepdim=True)*random[None]
            residual=max(residual,float(torch.einsum('btc,tcr->btr',edit,qp).abs().max()))
            norms[start:end] = torch.linalg.vector_norm(edit.flatten(1),dim=-1).cpu().numpy()
            source[start:end] = torch.linalg.vector_norm(h.flatten(1),dim=-1).cpu().numpy()
            with torch.inference_mode():
                future,context = continue_predict(model,h+edit)
            future,context = future.cpu().numpy(),context.cpu().numpy()
            arrays['predicted'][start:end] = readout(future,evaluators['predicted'])
            arrays['teacher'][start:end] = readout(future,evaluators['teacher'])
            arrays['context'][start:end] = readout(context,evaluators['context'])
        if kind=='zero':
            assert norms.max()==0
            for k in arrays:
                np.testing.assert_allclose(arrays[k],zero[k],atol=2e-3,rtol=1e-5)
        if residual > 1e-4:
            raise AssertionError(f'Position constraint residual: {residual}')
        atom_npz(dest,**arrays,indices=keep,ids=labels['ids'][keep],split=labels['split'][keep],
                 edit_norm=norms,source_norm=source,position_basis_residual=residual,seconds=time.monotonic()-tick)
        print(f'{num+1}/{len(conditions)} {name}: {time.monotonic()-tick:.1f}s',flush=True)
        save_json(args.output/'evaluation_progress.json',{'completed':num+1,'total':len(conditions),'last':name})
    summarize(args)


def summarize(args):
    labels = np.load(args.baseline/'labels.npz')
    zero = np.load(args.output/'conditions/zero.npz'); keep=zero['indices']
    times = (np.arange(8,16).reshape(4,2).mean(1)-7)/24
    tc = times-times.mean()
    result = {'splits':{}}
    for part in ('B','D'):
        mask = labels['split'][keep]==part
        result['splits'][part] = {}
        for size in SIZES:
            v = labels['velocity'][keep[mask]]
            expected = (size*v/np.linalg.norm(v,axis=-1,keepdims=True))@labels['world_to_pixel'][:2]
            unit = expected/np.linalg.norm(expected,axis=-1,keepdims=True)
            paths = {'spline':args.output/'conditions'/f'spline_{size:+.2f}.npz',
                     'random':args.output/'conditions'/f'random_{size:+.2f}.npz',
                     'linear':args.iterated/'conditions'/f'velocity_k032_{size:+.2f}.npz',
                     'linear_random':args.iterated/'conditions'/f'random_k032_{size:+.2f}.npz'}
            loaded={k:np.load(p) for k,p in paths.items()}
            for kind,z in loaded.items():
                np.testing.assert_array_equal(z['indices'],keep)
                change=(z['predicted']-zero['predicted'])[mask]
                slope=np.einsum('t,ntc->nc',tc,change)/(tc**2).sum()
                perpendicular=slope-(slope*unit).sum(1)[:,None]*unit
                item={'videos':int(mask.sum()),'delta_speed':size,
                      'predicted':effect_metrics(change,expected,times),
                      'teacher':effect_metrics((z['teacher']-zero['teacher'])[mask],expected,times),
                      'context_last_shift_pixels':bootstrap_mean(np.linalg.norm((z['context']-zero['context'])[mask,-1],axis=-1)),
                      'relative_edit_norm':bootstrap_mean((z['edit_norm']/z['source_norm'])[mask]),
                      'perpendicular_slope_pixels_per_second':bootstrap_mean(np.linalg.norm(perpendicular,axis=1)),
                      'position_basis_residual_max':float(z['position_basis_residual'])}
                if kind in ('spline','linear'):
                    control=loaded['random' if kind=='spline' else 'linear_random']
                    item['paired_vs_random']=effect_metrics((z['predicted']-control['predicted'])[mask],expected,times)
                if kind=='spline':
                    other=loaded['linear']
                    item['paired_vs_linear']=effect_metrics((z['predicted']-other['predicted'])[mask],expected,times)
                    item['paired_context_drift_difference']=bootstrap_mean(
                        np.linalg.norm((z['context']-zero['context'])[mask,-1],axis=1)-
                        np.linalg.norm((other['context']-zero['context'])[mask,-1],axis=1))
                result['splits'][part][f'{kind}_{size:+.2f}']=item
    save_json(args.output/'effects.json',result)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['fit','evaluate','summarize'])
    parser.add_argument('--output',type=Path,default=ROOT/'results/spline_velocity_v1')
    parser.add_argument('--work',type=Path,default=Path('/scratch/users/hmolin/jepa-interp/spline_velocity_v1'))
    parser.add_argument('--cache',type=Path,default=Path('/scratch/users/hmolin/jepa-interp/velocity_causal_v1'))
    parser.add_argument('--baseline',type=Path,default=ROOT/'results/velocity_causal_v1')
    parser.add_argument('--iterated',type=Path,default=ROOT/'results/iterated_velocity_v1')
    parser.add_argument('--iterated-work',type=Path,default=Path('/scratch/users/hmolin/jepa-interp/iterated_velocity_v1'))
    parser.add_argument('--batch-size',type=int,default=16)
    args=parser.parse_args(); configure()
    globals()[args.stage](args)


if __name__=='__main__':
    main()
