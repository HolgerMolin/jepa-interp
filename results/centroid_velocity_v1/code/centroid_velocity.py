"""Original pooled centroid-spline recipe lifted exactly to prefix-only tokens.

Uses the existing notebook helpers unchanged: A-fitted 64D PCA, empirical bin
centroids, natural cubic interpolation, and replacement of PCA coordinates.
Equal token shifts are the minimum-Frobenius-norm lift of a pooled change.
"""
import argparse
import itertools
import json
from pathlib import Path
import time

import numpy as np
from scipy.interpolate import PPoly
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
import torch

from utils import fit_centroid_spline, replace_pca_coordinates
from spline_velocity import sha
from iterated_velocity import atom_npz
from velocity_causal import ROOT, SEED, CHECKPOINT, configure, save_json, fit_readout, load_model, continue_predict
from velocity_intervention import readout, effect_metrics, bootstrap_mean

SIZES=[-.5,-.25,0.,.25,.5]
WORK=Path('/scratch/users/hmolin/jepa-interp')


def lift(h,pooled_delta):
    return h+pooled_delta[:,None,:]


def pooled_edit(features,target_coordinates,mean,components):
    return (target_coordinates-(features-mean)@components.T)@components


def fit(args):
    args.output.mkdir(exist_ok=True); args.work.mkdir(exist_ok=True)
    labels=np.load(args.baseline/'labels.npz'); split=labels['split']
    a,b,c=[np.flatnonzero(split==s) for s in ['A','B','C']]
    definition={
        'method':'original 64D pooled PCA / empirical speed-bin centroids / natural cubic interpolation / PCA coordinate replacement',
        'source_sha256':sha(__file__),'original_helpers_sha256':sha(ROOT/'utils.py'),
        'old_recipe':'experiments.ipynb Section5; unchanged utils.fit_centroid_spline and utils.replace_pca_coordinates',
        'necessary_changes':'prefix-only frames0-7 instead of full16-frame features; constant-across-token minimum-norm lift into the actual encoder',
        'fit':'A-only 64D PCA and speed-bin centroids',
        'selection':'B PCA reconstruction MSE at known speed; 8,12,16 uniform anchors with nearest-anchor bins',
        'intervention':'replace all top64 pooled PCA coordinates by f(target_speed); preserve pooled orthogonal complement and token deviations from the pooled mean',
        'lift':'add the same 1024D pooled delta to each of 1024 tokens; exact mean change and minimum total squared edit norm',
        'position':'no added position projection because that would alter the original replacement; measure position drift explicitly',
        'controls':'untouched zero; source-speed centroid replacement; fixed seeded orthogonal rotation of PCA edits preserving norms and source-target difference norms',
        'random_rotation_seed':SEED+203,'speed_changes_mps':SIZES,
        'evaluation':'same B/D clips with speeds in [.75,3.5]; preserve original direction as physical reference, without enforcing it in the edit',
        'effects':'report target minus untouched AND target minus source-speed replacement; never interpret source replacement as an identity',
        'linear_reference':'frozen iterated all-eligible linear method cap32; compare on identical cases',
        'pooled_speed_evaluator':'new prefix-appropriate ridge on C only; five-fold C-only alpha CV; diagnostic, never selects spline',
        'pooled_path_diagnostic':'original speed targets .25,2.125,4; 21 waypoints on all unordered pairs; spline versus endpoint chords only in pooled probe evaluation',
        'frozen_predictor_readouts':{k:sha(args.baseline/f'{k}_position_readout.npz') for k in ['predicted','teacher','context']},
        'labels_sha256':sha(args.baseline/'labels.npz'), 'fit_ids':labels['ids'][a].tolist(),
        'selection_ids':labels['ids'][b].tolist(),'speed_readout_ids':labels['ids'][c].tolist(),
        'history':'D previously inspected; exploratory transfer test, no new clips or post-outcome tuning',
    }
    path=args.output/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=definition:
        raise RuntimeError('Changed protocol: use a fresh output/work directory')
    save_json(path,definition)
    cache=args.work/'pooled.npy'
    if not cache.exists():
        tokens=np.load(args.cache/'layer19.npy',mmap_mode='r')
        pooled=np.empty((len(split),1024),dtype=np.float64)
        for start in range(0,len(split),32):
            pooled[start:start+32]=np.asarray(tokens[start:start+32]).mean(axis=1,dtype=np.float64)
        np.save(cache,pooled)
    pooled=np.load(cache)
    pca=PCA(n_components=64,svd_solver='full').fit(pooled[a])
    za,zb=pca.transform(pooled[a]),pca.transform(pooled[b])
    models={}; candidates=[]
    for count in [8,12,16]:
        curve,anchors,centroids,counts=fit_centroid_spline(za,labels['speed'][a],'speed',count,[.25,4.])
        mse=float(np.square(zb-curve(labels['speed'][b])).mean())
        candidates.append({'knots':count,'validation_mse':mse})
        models[count]=(curve,anchors,centroids,counts)
    best=min(candidates,key=lambda x:x['validation_mse'])
    curve,anchors,centroids,counts=models[best['knots']]
    np.testing.assert_allclose(curve(anchors),centroids,atol=1e-10)
    save_json(args.output/'selection.json',{'selected':best,'candidates':candidates,
              'pca_explained_variance_fraction':float(pca.explained_variance_ratio_.sum()),
              'bin_counts':counts.tolist(),'B_Amean_PCA_mse':float(np.square(zb).mean())})
    # Same single rotation for source and target: preserves paired increment norms too.
    rotation=np.linalg.qr(np.random.default_rng(SEED+203).normal(size=(64,64)))[0]
    atom_npz(args.output/'model.npz',pca_mean=pca.mean_,pca_components=pca.components_,
             spline_knots=curve.x,spline_coefficients=curve.c,anchors=anchors,centroids=centroids,
             bin_counts=counts,rotation=rotation,training_ids=labels['ids'][a],validation_ids=labels['ids'][b])
    coef,bias,alpha,cv=fit_readout(pooled,labels['speed'][:,None],c)
    atom_npz(args.output/'pooled_speed_readout.npz',coef=coef,intercept=bias,alpha=alpha,cv_mse=cv,training_ids=labels['ids'][c])
    # Verify helper-equivalent replacement and preservation outside PCA coordinates.
    sample=pooled[b[:16]]; target=curve(labels['speed'][b[:16]])
    delta=pooled_edit(sample,target,pca.mean_,pca.components_)
    np.testing.assert_allclose(sample+delta,replace_pca_coordinates(sample,pca,target),atol=1e-10)
    np.testing.assert_allclose(pca.transform(sample+delta),target,atol=1e-10)
    print('selected '+json.dumps(best),flush=True)
    weak={}
    for part in ['B','D']:
        ii=np.flatnonzero(split==part); features=pooled[ii]
        clean=(features@coef+bias)[:,0]
        paths=[]
        for start,end in itertools.combinations([.25,2.125,4.],2):
            fractions=np.linspace(0,1,21); speeds=start+fractions*(end-start)
            chord=(1-fractions[:,None])*curve(start)+fractions[:,None]*curve(end)
            per={}
            for name,coordinates in [('spline',curve(speeds)),('chord',chord)]:
                errs=[]
                for speed,point in zip(speeds,coordinates):
                    pred=(replace_pca_coordinates(features,pca,point)@coef+bias)[:,0]
                    errs.append(float(np.abs(pred-speed).mean()))
                per[name]={'interior_target_mae':float(np.mean(errs[1:-1])),'mae_by_waypoint':errs}
            paths.append({'start':start,'end':end,**per})
        weak[part]={'clean_mae':float(np.abs(clean-labels['speed'][ii]).mean()),
                    'clean_r2':float(r2_score(labels['speed'][ii],clean)), 'paths':paths,
                    'spline_interior_target_mae':float(np.mean([p['spline']['interior_target_mae'] for p in paths])),
                    'chord_interior_target_mae':float(np.mean([p['chord']['interior_target_mae'] for p in paths]))}
    save_json(args.output/'pooled_path_diagnostic.json',weak)


def evaluate(args):
    protocol=json.loads((args.output/'protocol.json').read_text())
    assert protocol['source_sha256']==sha(__file__)
    assert protocol['original_helpers_sha256']==sha(ROOT/'utils.py')
    for k,digest in protocol['frozen_predictor_readouts'].items():
        assert digest==sha(args.baseline/f'{k}_position_readout.npz')
    labels=np.load(args.baseline/'labels.npz')
    keep=np.flatnonzero(np.isin(labels['split'],['B','D']) & (labels['speed']>=.75) & (labels['speed']<=3.5))
    old=np.load(args.iterated/'conditions/zero.npz'); np.testing.assert_array_equal(old['indices'],keep)
    pooled=np.load(args.work/'pooled.npy'); m=np.load(args.output/'model.npz')
    curve=PPoly(m['spline_coefficients'],m['spline_knots'],extrapolate=False)
    basis,mean=m['pca_components'],m['pca_mean']; z=(pooled[keep]-mean)@basis.T
    speed_probe=np.load(args.output/'pooled_speed_readout.npz')
    conditions=[{'name':'zero','kind':'zero','size':0.}]+[
        {'name':f'{kind}_{size:+.2f}','kind':kind,'size':size} for size in SIZES for kind in ['centroid','random']]
    save_json(args.output/'evaluation_selection.json',{'conditions':conditions,'indices':keep.tolist(),
              'ids':labels['ids'][keep].tolist(),'model_sha256':sha(args.output/'model.npz'),
              'selection_sha256':sha(args.output/'selection.json'),
              'linear_baseline_sha256':{f'{size:+.2f}':sha(args.iterated/'conditions'/f'velocity_k032_{size:+.2f}.npz') for size in SIZES if size}})
    directory=args.output/'conditions'; directory.mkdir(exist_ok=True)
    evaluators={k:np.load(args.baseline/f'{k}_position_readout.npz') for k in ['predicted','teacher','context']}
    x=np.load(args.cache/'layer19.npy',mmap_mode='r')
    model=load_model(str(CHECKPOINT))
    qp=torch.tensor(np.load(WORK/'iterated_velocity_v1/position_basis.npy'),dtype=torch.float32,device='cuda')
    for num,condition in enumerate(conditions):
        name,kind,size=condition['name'],condition['kind'],condition['size']
        path=directory/f'{name}.npz'
        if path.exists():
            print('reuse '+name,flush=True); continue
        started=time.monotonic()
        target=labels['speed'][keep]+size
        assert target.min()>=.25-1e-10 and target.max()<=4.+1e-10
        coord_delta=curve(target)-z
        if kind=='zero': coord_delta=np.zeros_like(coord_delta)
        if kind=='random': coord_delta=coord_delta@m['rotation']
        delta=coord_delta@basis
        arrays={k:np.empty((len(keep),4,2)) for k in evaluators}
        norms=np.empty(len(keep));source=np.empty(len(keep)); pooled_speed=np.empty(len(keep))
        max_mean_error=0.;max_complement_error=0.;position_residual=0.
        for start in range(0,len(keep),args.batch_size):
            ii=keep[start:start+args.batch_size]; end=start+len(ii)
            h=torch.tensor(np.array(x[ii]),device='cuda')
            change=torch.tensor(delta[start:end],dtype=torch.float32,device='cuda')
            edited=lift(h,change)
            actual_mean=edited.double().mean(1).cpu().numpy()
            expected_mean=pooled[ii]+delta[start:end]
            max_mean_error=max(max_mean_error,float(abs(actual_mean-expected_mean).max()))
            actual_change=actual_mean-pooled[ii]
            max_complement_error=max(max_complement_error,float(abs(actual_change-(actual_change@basis.T)@basis).max()))
            if kind=='centroid':
                np.testing.assert_allclose((actual_mean-mean)@basis.T,curve(target[start:end]),atol=2e-5,rtol=1e-6)
            pooled_speed[start:end]=(actual_mean@speed_probe['coef']+speed_probe['intercept'])[:,0]
            position_residual=max(position_residual,float(torch.einsum('bc,tcr->btr',change,qp).abs().max()))
            norms[start:end]=np.sqrt(1024)*np.linalg.norm(delta[start:end],axis=-1)
            source[start:end]=torch.linalg.vector_norm(h.flatten(1),dim=-1).cpu().numpy()
            with torch.inference_mode(): future,context=continue_predict(model,edited)
            future,context=future.cpu().numpy(),context.cpu().numpy()
            arrays['predicted'][start:end]=readout(future,evaluators['predicted'])
            arrays['teacher'][start:end]=readout(future,evaluators['teacher'])
            arrays['context'][start:end]=readout(context,evaluators['context'])
        if kind=='zero':
            for k in arrays: np.testing.assert_allclose(arrays[k],old[k],atol=2e-3,rtol=1e-5)
        atom_npz(path,**arrays,indices=keep,ids=labels['ids'][keep],split=labels['split'][keep],
                 edit_norm=norms,source_norm=source,pooled_speed=pooled_speed,target_speed=target,
                 pooled_mean_max_error=max_mean_error,pooled_complement_max_error=max_complement_error,
                 position_basis_residual=position_residual,seconds=time.monotonic()-started)
        print(f'{num+1}/{len(conditions)} {name}: {time.monotonic()-started:.1f}s',flush=True)
        save_json(args.output/'evaluation_progress.json',{'completed':num+1,'total':len(conditions),'last':name})
    summarize(args)


def summary_item(z,reference,mask,expected,times):
    change=(z['predicted']-reference['predicted'])[mask]
    unit=expected/np.linalg.norm(expected,axis=1,keepdims=True)
    tc=times-times.mean(); slope=np.einsum('t,ntc->nc',tc,change)/(tc**2).sum()
    perpendicular=slope-(slope*unit).sum(1)[:,None]*unit
    return {'predicted':effect_metrics(change,expected,times),
            'teacher':effect_metrics((z['teacher']-reference['teacher'])[mask],expected,times),
            'context_last_shift_pixels':bootstrap_mean(np.linalg.norm((z['context']-reference['context'])[mask,-1],axis=1)),
            'perpendicular_slope_pixels_per_second':bootstrap_mean(np.linalg.norm(perpendicular,axis=1))}


def summarize(args):
    labels=np.load(args.baseline/'labels.npz'); zero=np.load(args.output/'conditions/zero.npz'); keep=zero['indices']
    m=np.load(args.output/'model.npz'); curve=PPoly(m['spline_coefficients'],m['spline_knots'],extrapolate=False)
    times=(np.arange(8,16).reshape(4,2).mean(1)-7)/24
    report={'splits':{}}
    for part in ['B','D']:
        mask=labels['split'][keep]==part; v=labels['velocity'][keep[mask]]
        result={'videos':int(mask.sum()),'source_replacement':{},'conditions':{}}
        for kind in ['centroid','random']:
            source=np.load(args.output/'conditions'/f'{kind}_+0.00.npz')
            truth=labels['target_position'][keep[mask]]
            result['source_replacement'][kind]={
                'context_last_shift_pixels':bootstrap_mean(np.linalg.norm((source['context']-zero['context'])[mask,-1],axis=1)),
                'future_last_shift_pixels':bootstrap_mean(np.linalg.norm((source['predicted']-zero['predicted'])[mask,-1],axis=1)),
                'relative_edit_norm':bootstrap_mean((source['edit_norm']/source['source_norm'])[mask]),
                'last_position_error_pixels':bootstrap_mean(np.linalg.norm(source['predicted'][mask,-1]-truth[:,-1],axis=1)),
                'pooled_speed_target_mae':bootstrap_mean(abs(source['pooled_speed'][mask]-labels['speed'][keep[mask]]))}
        for size in SIZES:
            if size==0: continue
            expected=(size*v/np.linalg.norm(v,axis=1,keepdims=True))@labels['world_to_pixel'][:2]
            linear=np.load(args.iterated/'conditions'/f'velocity_k032_{size:+.2f}.npz')
            np.testing.assert_array_equal(linear['indices'],keep)
            lin=summary_item(linear,zero,mask,expected,times)
            lin['relative_edit_norm']=bootstrap_mean((linear['edit_norm']/linear['source_norm'])[mask])
            result['conditions'][f'linear_{size:+.2f}']=lin
            for kind in ['centroid','random']:
                z=np.load(args.output/'conditions'/f'{kind}_{size:+.2f}.npz')
                anchor=np.load(args.output/'conditions'/f'{kind}_+0.00.npz')
                row={'from_original':summary_item(z,zero,mask,expected,times),
                     'from_source_replacement':summary_item(z,anchor,mask,expected,times),
                     'relative_edit_norm':bootstrap_mean((z['edit_norm']/z['source_norm'])[mask]),
                     'pooled_speed_target_mae':bootstrap_mean(abs(z['pooled_speed'][mask]-z['target_speed'][mask])),
                     'pooled_speed_change_gain':bootstrap_mean((z['pooled_speed'][mask]-anchor['pooled_speed'][mask])/size)}
                increment=(curve(labels['speed'][keep]+size)-curve(labels['speed'][keep]))@m['pca_components']
                row['source_target_relative_edit_norm']=bootstrap_mean((np.sqrt(1024)*np.linalg.norm(increment,axis=1)/z['source_norm'])[mask])
                if kind=='centroid':
                    control=np.load(args.output/'conditions'/f'random_{size:+.2f}.npz')
                    control_anchor=np.load(args.output/'conditions/random_+0.00.npz')
                    difference=((z['predicted']-anchor['predicted'])-(control['predicted']-control_anchor['predicted']))[mask]
                    row['paired_increment_vs_random']=effect_metrics(difference,expected,times)
                    row['paired_total_vs_linear']=effect_metrics((z['predicted']-linear['predicted'])[mask],expected,times)
                result['conditions'][f'{kind}_{size:+.2f}']=row
        report['splits'][part]=result
    save_json(args.output/'effects.json',report)


def main():
    p=argparse.ArgumentParser(); p.add_argument('stage',choices=['fit','evaluate','summarize'])
    p.add_argument('--output',type=Path,default=ROOT/'results/centroid_velocity_v1')
    p.add_argument('--work',type=Path,default=WORK/'centroid_velocity_v1')
    p.add_argument('--cache',type=Path,default=WORK/'velocity_causal_v1')
    p.add_argument('--baseline',type=Path,default=ROOT/'results/velocity_causal_v1')
    p.add_argument('--iterated',type=Path,default=ROOT/'results/iterated_velocity_v1')
    p.add_argument('--batch-size',type=int,default=16)
    args=p.parse_args();configure();globals()[args.stage](args)


if __name__=='__main__': main()
