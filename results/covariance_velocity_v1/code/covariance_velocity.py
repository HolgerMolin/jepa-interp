"""Regularized conditional second-moment transport around a centroid spline.

Uses existing clips. A fits; B selects Gaussian conditional likelihood only.
No downstream outcome or D score tunes the transport.
"""
import argparse
import json
import time
import numpy as np
import torch
from sklearn.decomposition import PCA
from iterated_centroid_velocity import contribution, difference, WORK, SIZES, kernel_candidates, kernel_predict
from diagnose_spline_residuals import curve_span, remove_span, scores
from velocity_causal import ROOT, SEED, CHECKPOINT, configure, save_json, load_model, continue_predict
from centroid_velocity import summary_item, lift
from velocity_intervention import readout, bootstrap_mean, effect_metrics
from iterated_velocity import atom_npz
from spline_velocity import sha

OUT=ROOT/'results/covariance_velocity_v1'
BASE=ROOT/'results/velocity_causal_v1'
PREVIOUS=ROOT/'results/iterated_centroid_velocity_v1'
LINEAR=ROOT/'results/iterated_velocity_v1'


def conditional_covariance(za,sa,queries,bandwidth,shrinkage):
    w=np.exp(-.5*((np.asarray(queries)[:,None]-sa[None])/bandwidth)**2)
    w/=w.sum(1,keepdims=True)
    cov=np.einsum('na,ai,aj->nij',w,za,za,optimize=True)
    return (1-shrinkage)*cov+shrinkage*np.eye(za.shape[1])[None]


def matrix_power(cov,power):
    value,vector=np.linalg.eigh(cov)
    if np.min(value)<=0:raise ValueError('Nonpositive covariance')
    return (vector*value[:,None,:]**power)@vector.swapaxes(-1,-2)


def transport(z,source_cov,target_cov):
    white=np.einsum('ni,nij->nj',z,matrix_power(source_cov,-.5))
    return np.einsum('ni,nij->nj',white,matrix_power(target_cov,.5))


def conditional_nll(z,cov):
    dim=cov.shape[-1];zactive=z[:,:dim]
    return .5*(np.linalg.slogdet(cov)[1]+np.einsum('ni,ni->n',zactive,np.linalg.solve(cov,zactive[...,None])[...,0])+np.square(z[:,dim:]).sum(1))


def spline_model():
    return dict(np.load(PREVIOUS/'models/round_001.npz'))


def fit():
    OUT.mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists():raise RuntimeError('Use a fresh output directory')
    lab=np.load(BASE/'labels.npz');x=np.load(WORK/'centroid_velocity_v1/pooled.npy');s=lab['speed']
    a,b,d=[np.flatnonzero(lab['split']==p) for p in ['A','B','D']]
    sm=spline_model();mean=x[a].mean(0);r=x-mean-contribution(sm,s)
    pca=PCA(n_components=64,svd_solver='full').fit(r[a]);z=pca.transform(r);scale=z[a].std(0);z/=scale
    save_json(OUT/'protocol.json',{'source_sha256':sha(__file__),
        'fit':'A only, original B-selected one-round centroid spline, residual PCA64 and coordinate scaling',
        'selection':'B conditional Gaussian NLL in the same 64 standardized dimensions; unmodified tail contributes identity NLL',
        'mean_model':'Original spline; residual mean assumed zero. Weighted second moments around this mean may include conditional mean error.',
        'covariance':'Gaussian kernel speed weights on A second moments, convex shrinkage to identity; no extrapolation',
        'dimensions':[8,16,32,64],'bandwidths':[.35,.7,1.4],'shrinkages':[.25,.5,.75,1.],
        'transport':'Symmetric source inverse square root, then target symmetric square root; preserve standardized residual coordinates',
        'edit':'Original mean-curve difference plus covariance-transport residual change; uniform lift to every token',
        'controls':'Frozen mean-only, frozen iterated linear, shape-only, seeded orthogonal rotation of combined edit',
        'deltas':SIZES,'random_rotation':'Same fixed 1024D rotation as preceding iterative centroid experiment',
        'evaluation':'Existing B/D 184 clips each, frozen predictor and readouts; D historically inspected, exploratory',
        'readout_sha256':{k:sha(BASE/f'{k}_position_readout.npz') for k in ['predicted','teacher','context']},
        'spline_sha256':sha(PREVIOUS/'models/round_001.npz'),
        'pooled_sha256':sha(WORK/'centroid_velocity_v1/pooled.npy')})
    rows=[]
    for dim in [8,16,32,64]:
        for bandwidth in [.35,.7,1.4]:
            for shrinkage in [.25,.5,.75,1.]:
                cov=conditional_covariance(z[a,:dim],s[a],s[b],bandwidth,shrinkage)
                row={'dim':dim,'bandwidth':bandwidth,'shrinkage':shrinkage,'B_nll':float(conditional_nll(z[b],cov).mean())}
                rows.append(row)
        print('fit dimension',dim,flush=True)
    selected=min(rows,key=lambda r:r['B_nll'])
    identity_nll=float(.5*np.square(z[b]).sum(1).mean())
    save_json(OUT/'selection.json',{'selected':selected,'candidates':rows,'B_identity_nll':identity_nll})
    atom_npz(OUT/'model.npz',components=pca.components_,residual_pca_mean=pca.mean_,scale=scale,mean=mean,
             za=z[a,:selected['dim']],sa=s[a],dim=selected['dim'],bandwidth=selected['bandwidth'],shrinkage=selected['shrinkage'])
    # A label-free span-removal quadratic audit confirms whether second-order
    # signal survives when true-speed subtraction is entirely avoided.
    sr=remove_span(x,mean,curve_span(sm));pc=PCA(n_components=64,svd_solver='full').fit(sr[a]);zz=pc.transform(sr)
    zz/=np.maximum(zz[a].std(0),1e-10);features={};settings={};auditrows={}
    for dim in [8,16,32,64]:
        i,j=np.triu_indices(dim);f=np.c_[zz[:,:dim],zz[:,i]*zz[:,j]]
        f=(f-f[a].mean(0))/np.maximum(f[a].std(0),1e-10);features[dim]=f
        sett,rr=kernel_candidates(f[a],f[b],s[a],s[b]);settings[dim]=sett['linear'];auditrows[dim]=rr
    bestdim=min(settings,key=lambda k:settings[k]['B_mse'])
    save_json(OUT/'span_quadratic_selection.json',{'selected_dim':bestdim,'settings':settings,'candidates':auditrows})
    audit={}
    for dim,f in features.items():
        pp=kernel_predict(f[a],f[np.r_[b,d]],s[a],settings[dim])
        audit[str(dim)]={'B':scores(pp[:len(b)],s[b]),'D':scores(pp[len(b):],s[d])}
    save_json(OUT/'span_quadratic_scores.json',audit)
    # Evaluate model likelihood only after selection is serialized.
    dd=selected['dim'];cov=conditional_covariance(z[a,:dd],s[a],s[d],selected['bandwidth'],selected['shrinkage'])
    nll={'D_selected_nll':float(conditional_nll(z[d],cov).mean()),'D_identity_nll':float(.5*np.square(z[d]).sum(1).mean())}
    save_json(OUT/'likelihood_audit.json',nll)
    print('Selected',selected,'identity',identity_nll,'D',nll,flush=True)


def edit(x,s,delta,model,sm):
    if np.any((s+delta<.25-1e-10)|(s+delta>4+1e-10)):raise ValueError('Outside support')
    dim=int(model['dim']);basis=model['components'][:dim];scale=model['scale'][:dim]
    residual=x-model['mean']-contribution(sm,s)-model['residual_pca_mean']
    z=(residual@basis.T)/scale
    cs=conditional_covariance(model['za'],model['sa'],s,float(model['bandwidth']),float(model['shrinkage']))
    ct=conditional_covariance(model['za'],model['sa'],s+delta,float(model['bandwidth']),float(model['shrinkage']))
    if delta==0:shape=np.zeros_like(x)
    else:shape=((transport(z,cs,ct)-z)*scale)@basis
    mean=difference([sm],s,delta)
    return mean+shape,shape


def evaluate():
    protocol=json.loads((OUT/'protocol.json').read_text());assert protocol['source_sha256']==sha(__file__)
    for k,digest in protocol['readout_sha256'].items():assert digest==sha(BASE/f'{k}_position_readout.npz')
    model=dict(np.load(OUT/'model.npz'));sm=spline_model();lab=np.load(BASE/'labels.npz')
    zero=np.load(PREVIOUS/'conditions/zero.npz');keep=zero['indices'];s=lab['speed'][keep]
    x=np.load(WORK/'centroid_velocity_v1/pooled.npy');rotation=np.load(PREVIOUS/'random_rotation.npy')
    tok=np.load(WORK/'velocity_causal_v1/layer19.npy',mmap_mode='r')
    readouts={k:np.load(BASE/f'{k}_position_readout.npz') for k in ['predicted','teacher','context']}
    speedprobe=np.load(ROOT/'results/centroid_velocity_v1/pooled_speed_readout.npz')
    conditions=[(kind,size) for size in SIZES for kind in ['combined','shape','random']]
    save_json(OUT/'evaluation_selection.json',{'conditions':conditions,'indices':keep.tolist(),'model_sha256':sha(OUT/'model.npz')})
    directory=OUT/'conditions';directory.mkdir(exist_ok=True)
    predictor=load_model(str(CHECKPOINT));checks={}
    # Validate transport identity, inverse and consistency at real source points.
    null,_=edit(x[keep],s,0.,model,sm);np.testing.assert_array_equal(null,np.zeros_like(null))
    delta,_=edit(x[keep],s,.25,model,sm);reverse,_=edit(x[keep]+delta,s+.25,-.25,model,sm)
    np.testing.assert_allclose(delta+reverse,0,atol=1e-9)
    checks['round_trip_max_error']=float(abs(delta+reverse).max());checks['zero_exact']=True
    for num,(kind,size) in enumerate(conditions):
        name=f'{kind}_{size:+.2f}';dest=directory/f'{name}.npz'
        if dest.exists():continue
        tick=time.monotonic();combined,shape=edit(x[keep],s,size,model,sm)
        change={'combined':combined,'shape':shape,'random':combined@rotation}[kind]
        if kind=='random':np.testing.assert_allclose(np.linalg.norm(change,axis=1),np.linalg.norm(combined,axis=1),atol=1e-10)
        arrays={k:np.empty((len(keep),4,2)) for k in readouts};source=np.empty(len(keep));pooled_speed=np.empty(len(keep));mean_error=0.
        for start in range(0,len(keep),16):
            ii=keep[start:start+16];end=start+len(ii)
            h=torch.tensor(np.array(tok[ii]),device='cuda');hh=lift(h,torch.tensor(change[start:end],dtype=torch.float32,device='cuda'))
            mean=hh.double().mean(1).cpu().numpy();mean_error=max(mean_error,float(abs(mean-x[ii]-change[start:end]).max()))
            source[start:end]=torch.linalg.vector_norm(h.flatten(1),dim=-1).cpu().numpy()
            pooled_speed[start:end]=(mean@speedprobe['coef']+speedprobe['intercept'])[:,0]
            with torch.inference_mode():future,context=continue_predict(predictor,hh)
            future=future.cpu().numpy();context=context.cpu().numpy()
            for k in ['predicted','teacher']:arrays[k][start:end]=readout(future,readouts[k])
            arrays['context'][start:end]=readout(context,readouts['context'])
        assert mean_error<2e-5
        atom_npz(dest,**arrays,indices=keep,ids=lab['ids'][keep],split=lab['split'][keep],
                 edit_norm=np.sqrt(1024)*np.linalg.norm(change,axis=1),source_norm=source,pooled_speed=pooled_speed,pooled_mean_max_error=mean_error)
        print(f'{num+1}/{len(conditions)} {name} {time.monotonic()-tick:.1f}s',flush=True)
    save_json(OUT/'checks.json',checks)
    summarize()


def summarize():
    lab=np.load(BASE/'labels.npz');zero=np.load(PREVIOUS/'conditions/zero.npz');keep=zero['indices']
    times=(np.arange(8,16).reshape(4,2).mean(1)-7)/24;report={}
    for split in ['B','D']:
        mask=lab['split'][keep]==split;v=lab['velocity'][keep[mask]];rows={}
        for size in SIZES:
            expected=(size*v/np.linalg.norm(v,axis=1,keepdims=True))@lab['world_to_pixel'][:2]
            mean=np.load(PREVIOUS/'conditions'/f'spline_k001_{size:+.2f}.npz')
            linear=np.load(LINEAR/'conditions'/f'velocity_k032_{size:+.2f}.npz')
            rand=np.load(OUT/'conditions'/f'random_{size:+.2f}.npz')
            for kind,z in [('mean',mean),('linear',linear)]+[(k,np.load(OUT/'conditions'/f'{k}_{size:+.2f}.npz')) for k in ['combined','shape','random']]:
                row=summary_item(z,zero,mask,expected,times)
                row['relative_edit_norm']=bootstrap_mean((z['edit_norm']/z['source_norm'])[mask])
                if 'pooled_speed' in z:row['pooled_speed_change_gain']=bootstrap_mean((z['pooled_speed']-zero['pooled_speed'])[mask]/size)
                if kind=='combined':
                    for key,ref in [('mean',mean),('linear',linear),('random',rand)]:row['paired_vs_'+key]=effect_metrics((z['predicted']-ref['predicted'])[mask],expected,times)
                rows[f'{kind}_{size:+.2f}']=row
        report[split]=rows
    save_json(OUT/'effects.json',report)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['fit','evaluate','summarize']);args=parser.parse_args()
    configure();globals()[args.stage]()
