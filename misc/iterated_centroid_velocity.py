"""Iterated centroid-spline residualization, frozen predictor test, residual audits.

Fit PCA and centroid curves on A residuals, choose knots and sequence prefix on B.
All edits sum curve differences, so zero requested change is an exact identity.
No nonlinear-nullspace or independence guarantee is assumed.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.interpolate import PPoly
from sklearn.decomposition import PCA
import torch

from utils import fit_centroid_spline
from misc.spline_velocity import sha
from misc.centroid_velocity import lift, summary_item
from misc.iterated_velocity import atom_npz
from misc.velocity_causal import ROOT, SEED, CHECKPOINT, configure, save_json, load_model, continue_predict
from misc.velocity_intervention import readout, bootstrap_mean, effect_metrics

WORK=Path('/scratch/users/hmolin/jepa-interp')
SIZES=[-.5,-.25,.25,.5]
MAX_ROUNDS=16
PATIENCE=3
MIN_GAIN=.001


def contribution(model,speed):
    speed=np.asarray(speed)
    if np.any((speed<.25-1e-10)|(speed>4.+1e-10)):
        raise ValueError('Spline extrapolation is forbidden')
    values=PPoly(model['coefficients'],model['knots'],extrapolate=False)(np.clip(speed,.25,4.))
    return (values-model['curve_mean'])@model['components']


def cumulative(models,speed):
    result=np.zeros((len(speed),models[0]['components'].shape[1]))
    for model in models: result+=contribution(model,speed)
    return result


def difference(models,speed,delta):
    return cumulative(models,speed+delta)-cumulative(models,speed)


def fit_sequence(xa,xb,sa,sb,max_rounds=MAX_ROUNDS,patience=PATIENCE,min_gain=MIN_GAIN,pca_dim=64,knot_options=(8,12,16)):
    """Only A/B inputs are accepted; all saved curves use original feature coordinates."""
    ra,rb=xa.copy(),xb.copy()
    initial=float(np.square(rb-ra.mean(0)).mean())
    significant_best=initial; stale=0
    models,history=[],[]
    for step in range(max_rounds):
        pca=PCA(n_components=pca_dim,svd_solver='full').fit(ra)
        za,zb=pca.transform(ra),pca.transform(rb)
        candidates=[]; fitted={}
        for knots in knot_options:
            curve,anchors,centroids,counts=fit_centroid_spline(za,sa,'speed',knots,[.25,4.])
            candidates.append({'knots':knots,'B_PCA_reconstruction_mse':float(np.square(zb-curve(sb)).mean())})
            fitted[knots]=(curve,anchors,centroids,counts)
        selected=min(candidates,key=lambda z:z['B_PCA_reconstruction_mse'])
        curve,anchors,centroids,counts=fitted[selected['knots']]
        model={'components':pca.components_,'pca_mean':pca.mean_,'knots':curve.x,'coefficients':curve.c,
               'curve_mean':curve(sa).mean(0),'anchors':anchors,'centroids':centroids,'counts':counts}
        ca,cb=contribution(model,sa),contribution(model,sb)
        before=float(np.square(rb-ra.mean(0)).mean())
        ra-=ca;rb-=cb
        after=float(np.square(rb-ra.mean(0)).mean())
        meaningful=significant_best-after>=min_gain*initial
        if meaningful:significant_best=after;stale=0
        else:stale+=1
        models.append(model)
        item={'round':step+1,'selected':selected,'candidates':candidates,
              'A_residual_variance':float(np.square(ra-ra.mean(0)).mean()),
              'B_residual_mse':after,'B_round_reduction':before-after,
              'B_fraction_removed_vs_original':1-after/initial,
              'meaningful_validation_improvement':meaningful,'consecutive_stale_rounds':stale,
              'PCA_explained_variance':float(pca.explained_variance_ratio_.sum()),
              'A_contribution_norm_mean':float(np.linalg.norm(ca,axis=1).mean())}
        history.append(item)
        print(f'round {step+1}: knots={selected["knots"]}, B residual MSE={after:.7f}, reduction={before-after:+.7f}, stale={stale}',flush=True)
        if stale>=patience:break
    # Round zero can win, but the one-round and attempted-iteration conditions
    # are still reported as fixed diagnostics rather than silently discarded.
    best=int(np.argmin([initial]+[r['B_residual_mse'] for r in history]))
    return models,history,{'initial_B_residual_mse':initial,'selected_prefix':best,
                         'fitted_rounds':len(models),'stop_reason':'validation_patience' if stale>=patience else 'round_cap'}


def load_models(args):
    return [dict(np.load(p)) for p in sorted((args.output/'models').glob('round_*.npz'))]


def fit(args):
    args.output.mkdir(exist_ok=True);(args.output/'models').mkdir(exist_ok=True)
    labels=np.load(args.baseline/'labels.npz')
    a,b=[np.flatnonzero(labels['split']==s) for s in ['A','B']]
    protocol={
        'source_sha256':sha(__file__),'helper_sha256':sha(ROOT/'utils.py'),
        'method':'refit 64D PCA on A residuals, fit empirical centroid spline, subtract A-centered speed-dependent contribution, repeat',
        'knot_candidates':[8,12,16],'knot_selection':'B residual PCA-coordinate reconstruction MSE at known speed, original helper',
        'subtraction':'R_next = R - (curve(speed)-mean_A_curve) @ PCA_components; preserves A mean, not a projection',
        'max_rounds':MAX_ROUNDS,'patience':PATIENCE,'meaningful_gain_fraction_of_initial_B_mse':MIN_GAIN,
        'stopping':'three rounds without improving B full-feature residual MSE over the last meaningful best by at least 0.1% of initial B MSE',
        'prefix_selection':'minimum full-feature B residual MSE among all attempted prefixes, including zero',
        'downstream_prefixes':'1,2,4,8,16 up to attempted count, plus final attempted and B-selected positive prefix',
        'edit':'H + sum_j [f_j(source_speed+delta)-f_j(source_speed)] broadcast identically to each token; no centroid replacement',
        'position':'no extra projection; report independent position drift and perpendicular response',
        'controls':'untouched zero and fixed seeded 1024D orthogonal rotation of each cumulative difference; exactly norm-matched and uniform across tokens',
        'random_seed':SEED+305,'speed_changes':SIZES,
        'evaluation':'same B/D clips, original speeds [.75,3.5], frozen predictor/readouts, original direction as physical reference',
        'audit':'fresh linear and RBF kernel ridge on pooled residuals, A fit, B hyperparameters; record settings before D scoring; audit only, not a sequence-selection criterion',
        'audit_alphas':[1e-4,1e-3,.01,.1,1.,10.,100.,1000.],
        'audit_RBF_squared_distance_scales':[.25,.5,1.,2.,4.],
        'audit_label_use':'known speed is used in residual subtraction at A/B/D; supervised diagnostic, not a label-free erasure procedure',
        'fit_ids':labels['ids'][a].tolist(),'selection_ids':labels['ids'][b].tolist(),
        'pooled_features_sha256':sha(WORK/'centroid_velocity_v1/pooled.npy'),
        'labels_sha256':sha(args.baseline/'labels.npz'),
        'readouts':{k:sha(args.baseline/f'{k}_position_readout.npz') for k in ['predicted','teacher','context']},
        'history':'D historically inspected; fixed exploratory test, no downstream tuning'}
    dest=args.output/'protocol.json'
    if dest.exists():raise RuntimeError('Use a fresh output directory for a new fit')
    save_json(dest,protocol)
    x=np.load(WORK/'centroid_velocity_v1/pooled.npy')
    models,history,selection=fit_sequence(x[a],x[b],labels['speed'][a],labels['speed'][b])
    caps=sorted(set([k for k in [1,2,4,8,16] if k<=len(models)]+[len(models)]+([selection['selected_prefix']] if selection['selected_prefix'] else [])))
    selection.update(evaluated_prefixes=caps,history=history)
    for j,model in enumerate(models):atom_npz(args.output/'models'/f'round_{j+1:03d}.npz',**model)
    save_json(args.output/'selection.json',selection)
    rng=np.random.default_rng(SEED+305)
    rotation=np.linalg.qr(rng.normal(size=(1024,1024)))[0]
    np.save(args.output/'random_rotation.npy',rotation)
    # One-round difference must equal the prior original centroid curve difference.
    old=np.load(args.centroid/'model.npz');oldcurve=PPoly(old['spline_coefficients'],old['spline_knots'],extrapolate=False)
    speeds=np.linspace(.75,3.5,31)
    olddelta=(oldcurve(speeds+.25)-oldcurve(speeds))@old['pca_components']
    newdelta=difference(models[:1],speeds,.25)
    np.testing.assert_allclose(newdelta,olddelta,atol=1e-9,rtol=1e-8)
    np.testing.assert_array_equal(difference(models,speeds,0.),np.zeros((len(speeds),1024)))
    save_json(args.output/'fit_checks.json',{'one_round_matches_original_centroid_curve_difference':True,
              'one_round_max_error':float(abs(newdelta-olddelta).max()),'zero_difference_exact_identity':True})
    print('frozen sequence '+json.dumps({k:v for k,v in selection.items() if k!='history'}),flush=True)


def evaluate(args):
    protocol=json.loads((args.output/'protocol.json').read_text());assert protocol['source_sha256']==sha(__file__)
    for k,digest in protocol['readouts'].items():assert digest==sha(args.baseline/f'{k}_position_readout.npz')
    selection=json.loads((args.output/'selection.json').read_text());caps=selection['evaluated_prefixes'];models=load_models(args)
    labels=np.load(args.baseline/'labels.npz');keep=np.flatnonzero(np.isin(labels['split'],['B','D'])&(labels['speed']>=.75)&(labels['speed']<=3.5))
    old=np.load(args.iterated/'conditions/zero.npz');np.testing.assert_array_equal(old['indices'],keep)
    conditions=[{'name':'zero','kind':'zero','cap':0,'size':0.}]+[
        {'name':f'{kind}_k{cap:03d}_{size:+.2f}','kind':kind,'cap':cap,'size':size}
        for cap in caps for size in SIZES for kind in ['spline','random']]
    save_json(args.output/'evaluation_selection.json',{'conditions':conditions,'ids':labels['ids'][keep].tolist(),'indices':keep.tolist(),
              'selection_sha256':sha(args.output/'selection.json'),'model_sha256':{p.name:sha(p) for p in sorted((args.output/'models').glob('*.npz'))},
              'linear_reference_sha256':{str(s):sha(args.iterated/'conditions'/f'velocity_k032_{s:+.2f}.npz') for s in SIZES}})
    directory=args.output/'conditions';directory.mkdir(exist_ok=True)
    rotation=np.load(args.output/'random_rotation.npy');x=np.load(WORK/'velocity_causal_v1/layer19.npy',mmap_mode='r')
    pooled=np.load(WORK/'centroid_velocity_v1/pooled.npy')
    speed_probe=np.load(args.centroid/'pooled_speed_readout.npz')
    evaluators={k:np.load(args.baseline/f'{k}_position_readout.npz') for k in ['predicted','teacher','context']}
    model=load_model(str(CHECKPOINT))
    for num,condition in enumerate(conditions):
        name,kind,cap,size=[condition[k] for k in ['name','kind','cap','size']]
        dest=directory/f'{name}.npz'
        if dest.exists():print('reuse '+name,flush=True);continue
        tick=time.monotonic()
        delta=np.zeros((len(keep),1024)) if kind=='zero' else difference(models[:cap],labels['speed'][keep],size)
        if kind=='random':delta=delta@rotation
        arrays={k:np.empty((len(keep),4,2)) for k in evaluators};norms=np.sqrt(1024)*np.linalg.norm(delta,axis=1)
        source=np.empty(len(keep));decoded_speed=np.empty(len(keep));max_mean_error=0.
        for start in range(0,len(keep),16):
            ii=keep[start:start+16];end=start+len(ii)
            h=torch.tensor(np.array(x[ii]),device='cuda');change=torch.tensor(delta[start:end],dtype=torch.float32,device='cuda')
            edited=lift(h,change);mean=edited.double().mean(1).cpu().numpy()
            max_mean_error=max(max_mean_error,float(abs(mean-pooled[ii]-delta[start:end]).max()))
            decoded_speed[start:end]=(mean@speed_probe['coef']+speed_probe['intercept'])[:,0]
            source[start:end]=torch.linalg.vector_norm(h.flatten(1),dim=-1).cpu().numpy()
            with torch.inference_mode():future,context=continue_predict(model,edited)
            future,context=future.cpu().numpy(),context.cpu().numpy()
            arrays['predicted'][start:end]=readout(future,evaluators['predicted'])
            arrays['teacher'][start:end]=readout(future,evaluators['teacher'])
            arrays['context'][start:end]=readout(context,evaluators['context'])
        if kind=='zero':
            for k in arrays:np.testing.assert_allclose(arrays[k],old[k],atol=2e-3,rtol=1e-5)
        if max_mean_error>2e-5:raise AssertionError(max_mean_error)
        atom_npz(dest,**arrays,indices=keep,ids=labels['ids'][keep],split=labels['split'][keep],edit_norm=norms,source_norm=source,
                 pooled_speed=decoded_speed,pooled_mean_max_error=max_mean_error,seconds=time.monotonic()-tick)
        print(f'{num+1}/{len(conditions)} {name}: {time.monotonic()-tick:.1f}s',flush=True)
        save_json(args.output/'evaluation_progress.json',{'completed':num+1,'total':len(conditions),'last':name})
    summarize(args)


def summarize(args):
    labels=np.load(args.baseline/'labels.npz');zero=np.load(args.output/'conditions/zero.npz');keep=zero['indices']
    caps=json.loads((args.output/'selection.json').read_text())['evaluated_prefixes']
    times=(np.arange(8,16).reshape(4,2).mean(1)-7)/24
    report={'splits':{}}
    for part in ['B','D']:
        mask=labels['split'][keep]==part;v=labels['velocity'][keep[mask]];result={}
        for size in SIZES:
            expected=(size*v/np.linalg.norm(v,axis=1,keepdims=True))@labels['world_to_pixel'][:2]
            linear=np.load(args.iterated/'conditions'/f'velocity_k032_{size:+.2f}.npz')
            result[f'linear_{size:+.2f}']=summary_item(linear,zero,mask,expected,times)
            result[f'linear_{size:+.2f}']['relative_edit_norm']=bootstrap_mean((linear['edit_norm']/linear['source_norm'])[mask])
            first=np.load(args.output/'conditions'/f'spline_k001_{size:+.2f}.npz')
            for cap in caps:
                for kind in ['spline','random']:
                    name=f'{kind}_k{cap:03d}_{size:+.2f}';z=np.load(args.output/'conditions'/f'{name}.npz')
                    item=summary_item(z,zero,mask,expected,times)
                    item['relative_edit_norm']=bootstrap_mean((z['edit_norm']/z['source_norm'])[mask])
                    item['pooled_speed_change_gain']=bootstrap_mean((z['pooled_speed']-zero['pooled_speed'])[mask]/size)
                    if kind=='spline':
                        random=np.load(args.output/'conditions'/f'random_k{cap:03d}_{size:+.2f}.npz')
                        for key,reference in [('random',random),('one_round',first),('linear',linear)]:
                            item[f'paired_vs_{key}']=effect_metrics((z['predicted']-reference['predicted'])[mask],expected,times)
                    result[name]=item
        report['splits'][part]=result
    save_json(args.output/'effects.json',report)


def centered_kernel(kaa,kxa):
    return kaa-kaa.mean(0)[None]-kaa.mean(1)[:,None]+kaa.mean(), kxa-kxa.mean(1)[:,None]-kaa.mean(0)[None]+kaa.mean()


def kernel_candidates(xa,xb,sa,sb):
    """Select linear/RBF ridge using A fit and B prediction error only."""
    mean=xa.mean(0);a=torch.tensor(xa-mean,dtype=torch.float64,device='cuda');b=torch.tensor(xb-mean,dtype=torch.float64,device='cuda')
    ya=torch.tensor(sa-sa.mean(),dtype=torch.float64,device='cuda');yb=torch.tensor(sb,dtype=torch.float64,device='cuda')
    aa=a@a.T/a.shape[1];ba=b@a.T/a.shape[1]
    da=(aa.diagonal()[:,None]+aa.diagonal()[None]-2*aa).clamp_min(0)
    db=(b.square().mean(1)[:,None]+aa.diagonal()[None]-2*ba).clamp_min(0)
    median=float(torch.median(da[torch.triu_indices(len(a),len(a),1,device='cuda').unbind()]))
    rows=[];chosen={}
    for kind,factor in [('linear',None)]+[('rbf',f) for f in [.25,.5,1.,2.,4.]]:
        ka,kb=(aa,ba) if kind=='linear' else (torch.exp(-da/(median*factor)),torch.exp(-db/(median*factor)))
        kc,bc=centered_kernel(ka,kb)
        eigen,u=torch.linalg.eigh(kc);eigen=eigen.clamp_min(0);rhs=u.T@ya;left=bc@u
        for alpha in [1e-4,1e-3,.01,.1,1.,10.,100.,1000.]:
            pred=left@(rhs/(eigen+alpha))+sa.mean();mse=float((pred-yb).square().mean())
            row={'kind':kind,'factor':factor,'alpha':alpha,'B_mse':mse,'A_median_squared_distance':median}
            rows.append(row)
            if kind not in chosen or mse<chosen[kind]['B_mse']:chosen[kind]=row
    return chosen,rows


def kernel_predict(xa,xt,sa,setting):
    mean=xa.mean(0);a=torch.tensor(xa-mean,dtype=torch.float64,device='cuda');x=torch.tensor(xt-mean,dtype=torch.float64,device='cuda')
    aa=a@a.T/a.shape[1];xa_kernel=x@a.T/a.shape[1]
    if setting['kind']=='rbf':
        da=(aa.diagonal()[:,None]+aa.diagonal()[None]-2*aa).clamp_min(0)
        dx=(x.square().mean(1)[:,None]+aa.diagonal()[None]-2*xa_kernel).clamp_min(0)
        width=setting['A_median_squared_distance']*setting['factor']
        aa,xa_kernel=torch.exp(-da/width),torch.exp(-dx/width)
    aa,xa_kernel=centered_kernel(aa,xa_kernel)
    y=torch.tensor(sa-sa.mean(),dtype=torch.float64,device='cuda')
    weights=torch.linalg.solve(aa+setting['alpha']*torch.eye(len(a),dtype=torch.float64,device='cuda'),y)
    return (xa_kernel@weights+sa.mean()).cpu().numpy()


def audit(args):
    labels=np.load(args.baseline/'labels.npz');x=np.load(WORK/'centroid_velocity_v1/pooled.npy');models=load_models(args)
    selection=json.loads((args.output/'selection.json').read_text());caps=[0]+selection['evaluated_prefixes']
    aa,bb,dd=[np.flatnonzero(labels['split']==part) for part in ['A','B','D']]
    chosen={};residuals={};allrows={}
    for cap in caps:
        # Construct only A/B residuals before fitting/selecting any audit models.
        ab=np.r_[aa,bb];r=x[ab].copy()
        if cap:r-=cumulative(models[:cap],labels['speed'][ab])
        settings,rows=kernel_candidates(r[:len(aa)],r[len(aa):],labels['speed'][aa],labels['speed'][bb])
        chosen[str(cap)]=settings;allrows[str(cap)]=rows;residuals[cap]=r
        print(f'residual audit selected kernels at round {cap}',flush=True)
    save_json(args.output/'audit_selection.json',{'chosen':chosen,'candidates':allrows,'fit_ids':labels['ids'][aa].tolist(),'selection_ids':labels['ids'][bb].tolist()})
    results={};predictions={}
    for cap in caps:
        r=residuals[cap];rd=x[dd].copy()
        if cap:rd-=cumulative(models[:cap],labels['speed'][dd])
        results[str(cap)]={}
        for kind,setting in chosen[str(cap)].items():
            pred=kernel_predict(r[:len(aa)],np.r_[r[len(aa):],rd],labels['speed'][aa],setting)
            predictions[f'k{cap:03d}_{kind}']=pred
            scores={}
            for part,sl,ii in [('B',slice(0,len(bb)),bb),('D',slice(len(bb),None),dd)]:
                truth=labels['speed'][ii];pp=pred[sl]
                scores[part]={'r2':float(1-np.square(pp-truth).sum()/np.square(truth-truth.mean()).sum()),'mae_mps':float(abs(pp-truth).mean())}
            results[str(cap)][kind]=scores
    save_json(args.output/'residual_audit.json',results)
    atom_npz(args.output/'audit_predictions.npz',**predictions,indices=np.r_[bb,dd],ids=labels['ids'][np.r_[bb,dd]])


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['fit','evaluate','summarize','audit'])
    p.add_argument('--output',type=Path,default=ROOT/'results/iterated_centroid_velocity_v1')
    p.add_argument('--baseline',type=Path,default=ROOT/'results/velocity_causal_v1')
    p.add_argument('--centroid',type=Path,default=ROOT/'results/centroid_velocity_v1')
    p.add_argument('--iterated',type=Path,default=ROOT/'results/iterated_velocity_v1')
    args=p.parse_args();configure();globals()[args.stage](args)


if __name__=='__main__':main()
