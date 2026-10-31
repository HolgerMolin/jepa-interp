"""Diagnose remaining speed information without using D for model selection."""
import json
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from iterated_centroid_velocity import contribution, kernel_candidates, kernel_predict, WORK
from velocity_causal import ROOT, SEED, configure, save_json
from iterated_velocity import atom_npz
from spline_velocity import sha

OUT=ROOT/'results/spline_residual_diagnostics_v1'
BASE=ROOT/'results/velocity_causal_v1'


def curve_span(model):
    centers=(model['centroids']-model['curve_mean'])@model['components']
    _,s,v=np.linalg.svd(centers,full_matrices=False)
    return v[s>s[0]*1e-10].T


def remove_span(x,mean,q):
    z=x-mean
    return z-(z@q)@q.T


def scores(pred,truth):
    return {'r2':float(1-np.square(pred-truth).sum()/np.square(truth-truth.mean()).sum()),
            'mae_mps':float(abs(pred-truth).mean())}


def main():
    configure();OUT.mkdir(exist_ok=True)
    if (OUT/'selection.json').exists():raise RuntimeError('Use a new output directory')
    lab=np.load(BASE/'labels.npz');x=np.load(WORK/'centroid_velocity_v1/pooled.npy');s=lab['speed']
    a,b,d=[np.flatnonzero(lab['split']==p) for p in ['A','B','D']]
    model=dict(np.load(ROOT/'results/iterated_centroid_velocity_v1/models/round_001.npz'))
    q=curve_span(model);mean=x[a].mean(0);r=x-contribution(model,s)
    probe=np.load(ROOT/'results/centroid_velocity_v1/pooled_speed_readout.npz')
    estimated=np.clip((x@probe['coef']+probe['intercept'])[:,0],.25,4.)
    rp=x-contribution(model,estimated)
    spanless=remove_span(x,mean,q)
    variants={'original':x,'true_speed_residual':r,'estimated_speed_residual':rp,'curve_span_removed':spanless}
    for name,features in [('residual',r),('span_removed',spanless),('original',x)]:
        z=features-features[a].mean(0);norm=np.linalg.norm(z,axis=1)
        variants[name+'_norm']=norm[:,None]
        variants[name+'_unit']=z/np.maximum(norm[:,None],1e-12)
    # Assigned speed remains the same; feature donors are independently permuted
    # inside each split. This deliberately destroys the feature-label pairing.
    for seed in range(3):
        rng=np.random.default_rng(SEED+410+seed);perm=np.arange(len(s))
        for ii in [a,b,d]:perm[ii]=rng.permutation(ii)
        variants[f'shuffled_{seed}']=x[perm]
        variants[f'shuffled_minus_curve_{seed}']=x[perm]-contribution(model,s)
    selected={};allrows={};preds={};results={}
    save_json(OUT/'protocol.json',{'source_sha256':sha(__file__),'seed':SEED+410,
        'fit':'A','selection':'B speed MSE','evaluation':'D exploratory; historically inspected',
        'curve_span_rank':q.shape[1], 'estimated_speed_source':'Frozen C-only pooled speed ridge; inference uses no speed labels',
        'span_projection':'Fixed orthogonal complement of all centered spline centroids; no per-clip labels',
        'shuffle':'Three independent within-split feature permutations, predict assigned original labels, paired before/after label-dependent subtraction',
        'polynomial':'A-residual PCA; standardized linear, squares, or all pairwise quadratic products; dimension selected by B',
        'no_steering_selection':'These diagnostic outcomes do not select downstream steering based on D'})
    # Explicit moment features isolate whether second-order statistics suffice.
    pca=PCA(n_components=64,svd_solver='full').fit(r[a]);z=pca.transform(r)
    z/=np.maximum(z[a].std(0),1e-10)
    for dim in [8,16,32,64]:
        zz=z[:,:dim]
        i,j=np.triu_indices(dim)
        for kind,feat in [('squares',zz*zz),('quadratic',zz[:,i]*zz[:,j])]:
            feat=np.c_[zz,feat];feat=(feat-feat[a].mean(0))/np.maximum(feat[a].std(0),1e-10)
            variants[f'{kind}_{dim}']=feat
    for name,features in variants.items():
        settings,rows=kernel_candidates(features[a],features[b],s[a],s[b])
        if name.startswith(('squares_','quadratic_')):settings={'linear':settings['linear']}
        selected[name]=settings;allrows[name]=rows
        print(name, {k:round(v['B_mse'],5) for k,v in settings.items()},flush=True)
    for kind in ['squares','quadratic']:
        names=[f'{kind}_{dim}' for dim in [8,16,32,64]]
        winner=min(names,key=lambda n:selected[n]['linear']['B_mse'])
        selected[kind+'_selected']={'variant':winner}
    save_json(OUT/'selection.json',{'selected':selected,'candidates':allrows})
    for name,features in variants.items():
        results[name]={}
        for kind,setting in selected[name].items():
            pp=kernel_predict(features[a],features[np.r_[b,d]],s[a],setting)
            preds[name+'_'+kind]=pp
            results[name][kind]={'B':scores(pp[:len(b)],s[b]),'D':scores(pp[len(b):],s[d])}
    save_json(OUT/'scores.json',results)
    atom_npz(OUT/'predictions.npz',**preds,indices=np.r_[b,d])
    # Verify the projection really removes the entire spline at unseen speeds.
    grid=np.linspace(.25,4,257);c=contribution(model,grid)
    err=float(abs(c-(c@q)@q.T).max())
    assert err<1e-9
    save_json(OUT/'checks.json',{'curve_span_rank':q.shape[1],'curve_projection_max_error':err,
        'estimated_speed_D':scores(estimated[d],s[d]),'all_predictions_finite':all(np.isfinite(p).all() for p in preds.values())})
    print(json.dumps({n:{k:v['D'] for k,v in z.items()} for n,z in results.items()},indent=2),flush=True)


if __name__=='__main__':main()
