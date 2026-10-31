"""Assemble, evaluate, and audit the frozen iterative velocity probe sequences."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from misc.iterated_velocity import CAPS, RANK_CAP, atom_npz
from misc.velocity_causal import CHECKPOINT, ROOT, SEED, configure, continue_predict, load_model, save_json
from misc.velocity_intervention import bootstrap_mean, effect_metrics, readout


def assemble(args):
    paths = sorted((args.work/'chunks').glob('tokens_*.npz'))
    all_indices = np.concatenate([np.load(p)['token_indices'] for p in paths])
    np.testing.assert_array_equal(all_indices, np.arange(1024))
    arrays = {}
    for key, shape in {'mean': (1024,1024), 'position_basis': (1024,1024,2),
                       'all_basis': (1024,1024,RANK_CAP),
                       'eligible_basis': (1024,1024,RANK_CAP),
                       'random_basis': (1024,1024,RANK_CAP),
                       'operators': (len(CAPS),1024,1024,2)}.items():
        arrays[key] = np.lib.format.open_memmap(args.work/f'{key}.npy', mode='w+', dtype=np.float64, shape=shape)
    counts = np.zeros(1024, dtype=int)
    ranks = np.zeros(1024, dtype=int)
    reasons = np.empty(1024, dtype='<U24')
    rounds = np.zeros(1024, dtype=int)
    terminal_r2 = np.zeros((1024,2))
    checks = {'constraint_max_error': 0., 'direct_ridge_max_difference': 0., 'basis_orthogonality_max_error': 0.}
    rng = np.random.default_rng(SEED+101)
    history_rows = []
    for path in paths:
        z = np.load(path)
        ii = z['token_indices']
        arrays['mean'][ii] = z['mean']
        arrays['position_basis'][ii] = z['position_basis']
        arrays['all_basis'][ii] = z['velocity_basis']
        ranks[ii] = z['final_rank']; reasons[ii] = z['stop_reason']
        counts[ii] = z['eligible'].sum(0); rounds[ii] = z['active'].sum(0)
        for t, index in enumerate(ii):
            terminal_r2[index] = z['validation_r2'][rounds[index]-1,t]
            selected = np.flatnonzero(z['eligible'][:,t])
            offset = 0
            op = np.zeros((1024,2))
            weights = []
            cumulative = [op.copy()]
            for step in selected:
                start = z['rank_before'][step,t]
                added = z['removed_rank'][step,t]
                assert added == 2
                arrays['eligible_basis'][index,:,offset:offset+2] = z['velocity_basis'][t,:,start:start+2]
                w = z['coef'][step,t]
                op += np.linalg.pinv(w, rcond=1e-8)
                cumulative.append(op.copy())
                weights.append(w)
                offset += 2
            for c, cap in enumerate(CAPS):
                num = min(cap,len(selected))
                arrays['operators'][c,index] = cumulative[num]
                if num:
                    residual = np.stack(weights[:num]) @ cumulative[num] - np.eye(2)
                    checks['constraint_max_error'] = max(checks['constraint_max_error'],float(abs(residual).max()))
            pc = z['position_basis'][t].T @ op
            checks['constraint_max_error'] = max(checks['constraint_max_error'],float(abs(pc).max()))
            for s in np.flatnonzero(z['active'][:,t]):
                history_rows.append([int(index),int(s),int(z['rank_before'][s,t]),
                                     *z['validation_r2'][s,t].tolist(),bool(z['eligible'][s,t]),
                                     float(z['alpha'][s,t])])
        checks['direct_ridge_max_difference'] = max(checks['direct_ridge_max_difference'],float(z['direct_check_max_difference']))
        checks['basis_orthogonality_max_error'] = max(checks['basis_orthogonality_max_error'],float(z['orthogonality_max_error']))
        # Nested random bases allow fair per-token rank matching at every cap.
        random = torch.tensor(rng.normal(size=(len(ii),1024,RANK_CAP)),dtype=torch.float64,device='cuda')
        qp = torch.as_tensor(z['position_basis'],dtype=torch.float64,device='cuda')
        random -= qp @ (qp.transpose(-1,-2) @ random)
        qrand = torch.linalg.qr(random,mode='reduced').Q
        arrays['random_basis'][ii] = qrand.cpu().numpy()
        print(f'assembled tokens through {ii[-1]+1}',flush=True)
    if checks['constraint_max_error'] > 1e-5:
        raise AssertionError(checks)
    for value in arrays.values():
        value.flush()
    # Do not repeat a cap once every token has already used all eligible probes.
    unique_caps = []
    previous = None
    for cap in CAPS:
        effective = np.minimum(counts,cap)
        if previous is None or not np.array_equal(previous,effective):
            unique_caps.append(cap)
        previous = effective
    atom_npz(args.output/'token_counts.npz', eligible_counts=counts, terminal_rank=ranks,
             stop_reason=reasons, fitted_probes=rounds, terminal_validation_r2=terminal_r2)
    atom_npz(args.output/'probe_history.npz', data=np.array(history_rows),
             columns=np.array(['token','iteration','rank_before','B_r2_vx','B_r2_vy','eligible','alpha']))
    summary = {'tokens':1024,'eligible_probe_count_quantiles':np.quantile(counts,[0,.1,.25,.5,.75,.9,1]).tolist(),
               'terminal_rank_quantiles':np.quantile(ranks,[0,.1,.25,.5,.75,.9,1]).tolist(),
               'stop_reasons':{str(k):int(v) for k,v in zip(*np.unique(reasons,return_counts=True))},
               'terminal_r2_component_quantiles':np.quantile(terminal_r2,[.1,.5,.9],axis=0).tolist(),
               'unique_caps':unique_caps,'checks':checks,
               'cap_distributions':{str(cap):{'probes_min':int(np.minimum(counts,cap).min()),
                    'probes_median':float(np.median(np.minimum(counts,cap))),
                    'probes_max':int(np.minimum(counts,cap).max()),
                    'mean_edited_dimensions':float(2*np.minimum(counts,cap).mean())} for cap in CAPS}}
    save_json(args.output/'sequence_summary.json',summary)


def erase(h,mean,q):
    """h [video,token,channel], q [token,channel,rank]."""
    centered = (h-mean).transpose(0,1)
    removed = (centered @ q) @ q.transpose(-1,-2)
    return h-removed.transpose(0,1)


def condition_list(caps):
    result = [{'name':'zero','kind':'zero','cap':0,'size':0.}]
    for cap in caps:
        for size in (-.5,-.25,.25,.5):
            for kind in ('velocity','random'):
                result.append({'name':f'{kind}_k{cap:03d}_{size:+.2f}','kind':kind,'cap':cap,'size':size})
        for kind in ('erase','random_erase'):
            result.append({'name':f'{kind}_k{cap:03d}','kind':kind,'cap':cap,'size':0.})
    result += [{'name':kind+'_all','kind':kind,'cap':-1,'size':0.} for kind in ('erase','random_erase')]
    return result


def evaluate(args):
    labels = np.load(args.baseline/'labels.npz')
    summary = json.loads((args.output/'sequence_summary.json').read_text())
    counts = np.load(args.output/'token_counts.npz')
    x = np.load(args.cache/'layer19.npy',mmap_mode='r')
    keep = np.flatnonzero(np.isin(labels['split'],['B','D']) & (labels['speed']>=.75) & (labels['speed']<=3.5))
    evaluators = {k:np.load(args.baseline/f'{k}_position_readout.npz') for k in ('predicted','teacher','context')}
    conditions = condition_list(summary['unique_caps'])
    save_json(args.output/'evaluation_selection.json',{'conditions':conditions,'ids':labels['ids'][keep].tolist(),
              'splits':labels['split'][keep].tolist(),'note':'all caps fixed by B probe eligibility before downstream effects',
              'evaluation_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'frozen_readout_sha256':{k:hashlib.sha256((args.baseline/f'{k}_position_readout.npz').read_bytes()).hexdigest()
                                       for k in evaluators}})
    directory = args.output/'conditions'; directory.mkdir(exist_ok=True)
    operator_array = np.load(args.work/'operators.npy',mmap_mode='r')
    bases = {k:np.load(args.work/f'{k}_basis.npy',mmap_mode='r') for k in ('eligible','all','random','position')}
    mean = torch.tensor(np.load(args.work/'mean.npy'),dtype=torch.float32,device='cuda')
    rng = np.random.default_rng(SEED+102)
    random = torch.tensor(rng.normal(size=(1024,1024)),dtype=torch.float32,device='cuda')
    qp = torch.tensor(np.array(bases['position']),dtype=torch.float32,device='cuda')
    random -= (qp @ (qp.transpose(-1,-2) @ random[:,:,None])).squeeze(-1)
    random /= torch.linalg.vector_norm(random,dim=-1,keepdim=True)
    model = load_model(str(CHECKPOINT))
    started = time.monotonic()
    previous = np.load(args.baseline/'baseline_predictions.npz')
    for cnum,condition in enumerate(conditions):
        name,kind,cap,size = [condition[k] for k in ('name','kind','cap','size')]
        path = directory/f'{name}.npz'
        if path.exists():
            print(f'reuse {name}',flush=True); continue
        tick = time.monotonic()
        op = q = None
        if kind in ('velocity','random'):
            op = torch.tensor(np.array(operator_array[CAPS.index(cap)]),dtype=torch.float32,device='cuda')
        elif 'erase' in kind:
            ranks = counts['terminal_rank'] if cap == -1 else 2*np.minimum(counts['eligible_counts'],cap)
            key = 'random' if kind=='random_erase' else 'all' if cap==-1 else 'eligible'
            width = int(ranks.max())
            q = torch.tensor(np.array(bases[key][:,:,:width]),dtype=torch.float32,device='cuda')
            mask = torch.arange(width,device='cuda')[None] < torch.tensor(ranks,device='cuda')[:,None]
            q *= mask[:,None]
        arrays = {key:np.empty((len(keep),4,2)) for key in ('predicted','teacher','context')}
        norms = np.empty(len(keep)); source_norms = np.empty(len(keep)); position_residual = 0.
        for start in range(0,len(keep),args.batch_size):
            ii = keep[start:start+args.batch_size]; end = start+len(ii)
            h = torch.tensor(np.array(x[ii]),device='cuda')
            if kind in ('velocity','random'):
                v = labels['velocity'][ii]
                delta = size*v/np.linalg.norm(v,axis=-1,keepdims=True)
                edit = torch.einsum('tcd,bd->btc',op,torch.tensor(delta,dtype=torch.float32,device='cuda'))
                if kind=='random':
                    edit = np.sign(size)*torch.linalg.vector_norm(edit,dim=-1,keepdim=True)*random[None]
            elif 'erase' in kind:
                edit = erase(h,mean,q)-h
            else:
                edit = torch.zeros_like(h)
            position_residual = max(position_residual,float(torch.einsum('btc,tcr->btr',edit,qp).abs().max()))
            norms[start:end] = torch.linalg.vector_norm(edit.flatten(1),dim=-1).cpu().numpy()
            source_norms[start:end] = torch.linalg.vector_norm(h.flatten(1),dim=-1).cpu().numpy()
            with torch.inference_mode():
                future,context = continue_predict(model,h+edit)
            future,context = future.cpu().numpy(),context.cpu().numpy()
            arrays['predicted'][start:end] = readout(future,evaluators['predicted'])
            arrays['teacher'][start:end] = readout(future,evaluators['teacher'])
            arrays['context'][start:end] = readout(context,evaluators['context'])
        if kind=='zero':
            for key,old in [('predicted','predicted'),('teacher','teacher_on_predicted'),('context','context')]:
                np.testing.assert_allclose(arrays[key],previous[old][keep],atol=2e-3,rtol=1e-5)
        atom_npz(path,**arrays,indices=keep,ids=labels['ids'][keep],split=labels['split'][keep],
                 edit_norm=norms,source_norm=source_norms,position_basis_residual=position_residual,
                 seconds=time.monotonic()-tick)
        print(f'{cnum+1}/{len(conditions)} {name}: {time.monotonic()-tick:.1f}s; elapsed {(time.monotonic()-started)/60:.1f}m',flush=True)
        save_json(args.output/'evaluation_progress.json',{'completed':cnum+1,'total':len(conditions),'last':name})
    summarize_effects(args,labels,conditions,keep)


def summarize_effects(args,labels,conditions,keep):
    zero = np.load(args.output/'conditions/zero.npz')
    times = (np.arange(8,16).reshape(4,2).mean(1)-7)/24
    report = {'splits':{},'zero_edit_matches_baseline':True}
    for part in ('B','D'):
        mask = labels['split'][keep]==part
        report['splits'][part] = {}
        truth = labels['target_position'][keep[mask]]
        base_errors = np.linalg.norm(zero['predicted'][mask]-truth,axis=-1)
        for condition in conditions[1:]:
            name,kind,cap,size = [condition[k] for k in ('name','kind','cap','size')]
            z = np.load(args.output/'conditions'/f'{name}.npz')
            item = {'videos':int(mask.sum()),'kind':kind,'cap':cap,'delta_speed':size,
                    'relative_edit_norm':bootstrap_mean((z['edit_norm']/z['source_norm'])[mask]),
                    'position_basis_residual_max':float(z['position_basis_residual']),
                    'context_last_shift_pixels':bootstrap_mean(np.linalg.norm((z['context']-zero['context'])[mask,-1],axis=-1))}
            if kind in ('velocity','random'):
                v = labels['velocity'][keep[mask]]
                expected = (size*v/np.linalg.norm(v,axis=-1,keepdims=True)) @ labels['world_to_pixel'][:2]
                for key in ('predicted','teacher'):
                    item[key] = effect_metrics((z[key]-zero[key])[mask],expected,times)
                if kind=='velocity':
                    r = np.load(args.output/'conditions'/f'random_k{cap:03d}_{size:+.2f}.npz')
                    item['paired_vs_random'] = effect_metrics((z['predicted']-r['predicted'])[mask],expected,times)
            else:
                errors = np.linalg.norm(z['predicted'][mask]-truth,axis=-1)
                item['mean_position_error_pixels'] = errors.mean(0).tolist()
                item['position_error_increase_pixels'] = [bootstrap_mean(errors[:,t]-base_errors[:,t]) for t in range(4)]
                target_disp = truth-labels['positions'][keep[mask],7:8]
                predicted_disp = z['predicted'][mask]-labels['positions'][keep[mask],7:8]
                item['trajectory_displacement_gain'] = bootstrap_mean((predicted_disp*target_disp).sum((1,2))/(target_disp**2).sum((1,2)))
                if kind=='erase':
                    suffix = 'all' if cap==-1 else f'k{cap:03d}'
                    r = np.load(args.output/'conditions'/f'random_erase_{suffix}.npz')
                    re = np.linalg.norm(r['predicted'][mask]-truth,axis=-1)
                    item['paired_error_vs_random_pixels'] = [bootstrap_mean(errors[:,t]-re[:,t]) for t in range(4)]
            report['splits'][part][name] = item
    save_json(args.output/'effects.json',report)


@torch.inference_mode()
def audit(args):
    labels = np.load(args.baseline/'labels.npz')
    keep = np.flatnonzero(labels['split']!='C')
    split = labels['split'][keep]
    a = np.flatnonzero(split=='A'); b = np.flatnonzero(split=='B'); d = np.flatnonzero(split=='D')
    x = np.load(args.cache/'layer19.npy',mmap_mode='r')
    means = np.load(args.work/'mean.npy',mmap_mode='r')
    qv = np.load(args.work/'all_basis.npy',mmap_mode='r')
    qp = np.load(args.work/'position_basis.npy',mmap_mode='r')
    qr = np.load(args.work/'random_basis.npy',mmap_mode='r')
    ranks = np.load(args.output/'token_counts.npz')['terminal_rank']
    kernels = {key:torch.zeros((len(keep),len(a)),device='cuda',dtype=torch.float64)
               for key in ('original','velocity_erased','random_erased','position_and_velocity_erased')}
    for start in range(0,1024,16):
        end = start+16
        raw = torch.tensor(np.array(x[keep,start:end]),dtype=torch.float64,device='cuda')
        mean = torch.tensor(np.array(means[start:end]),dtype=torch.float64,device='cuda')
        raw -= mean
        va = torch.tensor(np.array(qv[start:end]),dtype=torch.float64,device='cuda')
        pa = torch.tensor(np.array(qp[start:end]),dtype=torch.float64,device='cuda')
        ra = torch.tensor(np.array(qr[start:end]),dtype=torch.float64,device='cuda')
        valid = torch.arange(RANK_CAP,device='cuda')[None] < torch.tensor(ranks[start:end],device='cuda')[:,None]
        ra *= valid[:,None]
        for key in kernels:
            if key=='original':
                h = raw
            elif key=='velocity_erased':
                h = erase(raw,0,va)
            elif key=='random_erased':
                h = erase(raw,0,ra)
            else:
                h = erase(erase(raw,0,va),0,pa)
            h = h.flatten(1)
            kernels[key].addmm_(h,h[a].T)
        print(f'joint-token Gram matrices {end}/1024',flush=True)
    y = labels['velocity'][keep]
    scale = y[a].std(0); mean_y = y[a].mean(0)
    centered = (y[a]-mean_y)/scale
    alpha_grid = np.logspace(-4,8,13)
    result = {}; predictions = {}; selected = {}
    for key,kernel in kernels.items():
        kernel = kernel.cpu().numpy()
        gram = kernel[a]
        values,vectors = np.linalg.eigh((gram+gram.T)*.5)
        if values.min() < -1e-7*values.max():
            raise AssertionError('Joint Gram is not positive semidefinite')
        values = np.maximum(values,0.)
        rhs = vectors.T @ centered
        val = []; candidates = []
        for alpha in alpha_grid:
            dual = vectors @ (rhs/(values[:,None]+alpha))
            pred = (kernel@dual)*scale+mean_y
            candidates.append(pred)
            val.append(float(np.abs((pred[b]-y[b])/scale).mean()))
        best = int(np.argmin(val)); predictions[key] = candidates[best]
        selected[key] = {'alpha':float(alpha_grid[best]),'validation_standardized_mae_grid':val,
                         'alpha_grid_edge':best in (0,len(alpha_grid)-1)}
    save_json(args.output/'joint_audit_selection.json',{'alphas':alpha_grid.tolist(),'selected':selected})
    for key,pred in predictions.items():
        result[key] = {'selection':selected[key]}
        for part,ix in [('B',b),('D',d)]:
            error = pred[ix]-y[ix]
            result[key][part] = {'r2_components':(1-(error**2).sum(0)/((y[ix]-y[ix].mean(0))**2).sum(0)).tolist(),
                                'mae_components_mps':np.abs(error).mean(0).tolist()}
    save_json(args.output/'joint_audit.json',result)
    atom_npz(args.output/'joint_audit_predictions.npz',**predictions,ids=labels['ids'][keep],split=split)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['assemble','evaluate','audit'])
    parser.add_argument('--baseline',type=Path,default=ROOT/'results/velocity_causal_v1')
    parser.add_argument('--cache',type=Path,default=Path('/scratch/users/hmolin/jepa-interp/velocity_causal_v1'))
    parser.add_argument('--output',type=Path,default=ROOT/'results/iterated_velocity_v1')
    parser.add_argument('--work',type=Path,default=Path('/scratch/users/hmolin/jepa-interp/iterated_velocity_v1'))
    parser.add_argument('--batch-size',type=int,default=2)
    args = parser.parse_args(); configure(); globals()[args.stage](args)


if __name__=='__main__':
    main()
