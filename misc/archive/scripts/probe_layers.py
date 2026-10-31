"""Fit A, select on B, freeze decisions, then evaluate on D; C remains reserved."""
import json, hashlib
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'; FEATURES=Path('/scratch/users/hmolin/jepa-interp/features')
ALPHAS=np.array([1e-4,1e-3,1e-2,.1,1,10,100,1000,10000],dtype=float)
SEED=20260929

def metrics(y,p,task):
    r2=float(np.mean(1-np.sum((y-p)**2,axis=0)/np.maximum(np.sum((y-y.mean(0))**2,axis=0),1e-15)))
    if task=='direction':
        angles=np.arctan2(p[:,0],p[:,1]); truth=np.arctan2(y[:,0],y[:,1])
        # Undefined near-zero vectors receive a deterministic angle of zero, not exclusion.
        angles[np.linalg.norm(p,axis=1)<1e-8]=0
        err=np.abs((angles-truth+np.pi)%(2*np.pi)-np.pi)*180/np.pi
    else: err=np.abs(y-p).ravel()
    return {'r2':r2,'mae':float(err.mean()),'near_zero_fraction':float(np.mean(np.linalg.norm(p,axis=1)<1e-8)) if task=='direction' else None}

def intervals(y,p,task):
    rng=np.random.default_rng(SEED); vals=[]
    for _ in range(500):
        ii=rng.integers(0,len(y),len(y)); m=metrics(y[ii],p[ii],task); vals.append([m['r2'],m['mae']])
    lo,hi=np.quantile(vals,[.025,.975],axis=0)
    return {'r2':[float(lo[0]),float(hi[0])],'mae':[float(lo[1]),float(hi[1])]}

def ridge_candidates(x,y):
    xm=x.mean(0); ym=y.mean(0); z=x-xm
    eig,u=eigh(z@z.T,check_finite=False); eig=np.maximum(eig,0)
    uy=u.T@(y-ym)
    ws=[z.T@(u@(uy/(eig[:,None]+a))) for a in ALPHAS]
    return xm,ym,ws,(eig,u,z)

def shuffled_candidates(y,state):
    eig,u,z=state; rng=np.random.default_rng(SEED); y=y[rng.permutation(len(y))]
    ym=y.mean(0); uy=u.T@(y-ym)
    return ym,[z.T@(u@(uy/(eig[:,None]+a))) for a in ALPHAS]

def self_test():
    from sklearn.linear_model import Ridge
    rng=np.random.default_rng(7); x=rng.normal(size=(35,12)); y=rng.normal(size=(35,2))
    xm,ym,ws,_=ridge_candidates(x,y)
    ref=Ridge(alpha=ALPHAS[4]).fit(x,y)
    np.testing.assert_allclose((x-xm)@ws[4]+ym,ref.predict(x),rtol=1e-8,atol=1e-8)
    angles=np.deg2rad([359,1]); yy=np.column_stack([np.sin(angles),np.cos(angles)])
    assert abs(metrics(yy,yy[::-1],'direction')['mae']-2)<1e-10
    print('Ridge and circular-metric tests passed',flush=True)

def main():
 self_test()
 records=json.loads((OUT/'splits.json').read_text())['records']
 assert len({r['sha256'] for r in records})==len(records)
 cfg={'seed':SEED,'alphas':ALPHAS.tolist(),'fit':'A','validation':'B','reserved_evaluator':'C','test':'D','layer_indexing':'1-based block residual outputs before final encoder LayerNorm','selection':'minimum validation MAE; common layer maximizes mean validation R2 if each task R2 >= 0.5, otherwise task-specific validation-MAE minima','bootstrap_replicates':500,'feature_scaling':'training mean centering only','controls':'fixed clean-trained probe on 64 label-stratified test clips under static/reordered input; sensitivity, not refitted baseline','splits_sha256':hashlib.sha256((OUT/'splits.json').read_bytes()).hexdigest()}
 (OUT/'probe_protocol.json').write_text(json.dumps(cfg,indent=2)+'\n')
 cache={}; selection={}
 for task in ['direction','speed','acceleration']:
    rr=[r for r in records if r['task']==task]
    pieces=[]; ids=[]
    for path in sorted(FEATURES.glob(f'{task}_[0-9]*.npz')):
        with np.load(path) as f: pieces.append(f['features']); ids.extend(f['ids'].tolist())
    x=np.concatenate(pieces).astype(np.float64)
    assert ids==[r['id'] for r in rr] and x.shape==(len(rr),24,1024) and np.isfinite(x).all()
    if task=='direction':
        t=np.deg2rad([r['metadata']['theta_degrees'] for r in rr]); y=np.column_stack([np.sin(t),np.cos(t)])
    else: y=np.array([r['metadata']['speed_mps' if task=='speed' else 'acceleration_mps2'] for r in rr])[:,None]
    masks={s:np.array([r['split']==s for r in rr]) for s in 'ABCD'}
    assert sum(int(m.sum()) for m in masks.values())==len(rr)
    a,b=masks['A'],masks['B']; models=[]; choices=[]
    for layer in range(24):
        xm,ym,ws,state=ridge_candidates(x[a,layer],y[a]); svm,sws=shuffled_candidates(y[a],state)
        vm=[metrics(y[b],(x[b,layer]-xm)@w+ym,task) for w in ws]
        sm=[metrics(y[b],(x[b,layer]-xm)@w+svm,task) for w in sws]
        best=int(np.argmin([v['mae'] for v in vm])); sbest=int(np.argmin([v['mae'] for v in sm]))
        models.append((xm,ym,ws[best],svm,sws[sbest])); choices.append({'layer':layer+1,'alpha':float(ALPHAS[best]),'validation':vm[best],'shuffled_alpha':float(ALPHAS[sbest])})
        print(task,'layer',layer+1,'validation',vm[best],flush=True)
    selection[task]={'layers':choices,'best_task_layer':1+int(np.argmin([v['validation']['mae'] for v in choices]))}
    cache[task]=(rr,x,y,masks,models)
 common=int(np.argmax(np.mean([[c['validation']['r2'] for c in selection[t]['layers']] for t in selection],axis=0)))+1
 use_common=all(selection[t]['layers'][common-1]['validation']['r2']>=.5 for t in selection)
 for t in selection: selection[t]['selected_layer']=common if use_common else selection[t]['best_task_layer']
 (OUT/'validation_selection.json').write_text(json.dumps({'common_layer_used':use_common,'tasks':selection},indent=2)+'\n')
 # No test score has been computed before the selection file is frozen above.
 results={}
 for task,(rr,x,y,masks,models) in cache.items():
    a,b,d=masks['A'],masks['B'],masks['D']; layers=[]
    for layer,(xm,ym,w,svm,sw) in enumerate(models):
        pred=(x[:,layer]-xm)@w+ym; sp=(x[d,layer]-xm)@sw+svm
        item={**selection[task]['layers'][layer], 'train':metrics(y[a],pred[a],task),'test':metrics(y[d],pred[d],task),'test_ci95':intervals(y[d],pred[d],task),'shuffled_test':metrics(y[d],sp,task)}
        layers.append(item)
        np.savez(OUT/f'probe_{task}_layer{layer+1:02d}.npz',feature_mean=xm,target_mean=ym,weight=w,alpha=item['alpha'],test_ids=np.array([r['id'] for r in rr])[d],test_prediction=pred[d],test_target=y[d])
    baseline=np.repeat(y[a].mean(0,keepdims=True),sum(d),axis=0)
    selected=selection[task]['selected_layer']; xm,ym,w,_,_=models[selected-1]; control={}
    lookup={r['id']:i for i,r in enumerate(rr)}
    for name in ['static','shuffle']:
        with np.load(FEATURES/f'{task}_control_{name}.npz') as f:
            ix=np.array([lookup[int(i)] for i in f['ids']]); assert d[ix].all()
            cp=(f['features'][:,selected-1].astype(float)-xm)@w+ym
        clean=(x[ix,selected-1]-xm)@w+ym
        control[name]={'n':len(ix),'clean':metrics(y[ix],clean,task),'perturbed':metrics(y[ix],cp,task),'paired_mae_change':metrics(y[ix],cp,task)['mae']-metrics(y[ix],clean,task)['mae']}
    results[task]={'selected_layer':selected,'constant_test':metrics(y[d],baseline,task),'layers':layers,'controls':control}
 (OUT/'probe_results.json').write_text(json.dumps(results,indent=2)+'\n')
 fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained')
 for col,(task,r) in enumerate(results.items()):
    for row,metric in enumerate(['r2','mae']):
        ax=axes[row,col]; ll=r['layers']; xx=np.arange(1,25)
        yy=[v['test'][metric] for v in ll]; ci=np.array([v['test_ci95'][metric] for v in ll])
        ax.plot(xx,yy,label='Test'); ax.fill_between(xx,ci[:,0],ci[:,1],alpha=.2,label='95% clip bootstrap')
        ax.plot(xx,[v['validation'][metric] for v in ll],'--',label='Validation')
        ax.plot(xx,[v['shuffled_test'][metric] for v in ll],':',label='Shuffled labels')
        ax.axhline(r['constant_test'][metric],color='gray',lw=1,label='Constant')
        ax.axvline(r['selected_layer'],color='black',alpha=.3)
        ax.set(xlabel='Transformer block (1-based)',ylabel='R²' if metric=='r2' else 'MAE ('+{'direction':'degrees','speed':'m/s','acceleration':'m/s²'}[task]+')',title=task)
        ax.grid(alpha=.2)
 axes[0,0].legend(fontsize=7)
 fig.savefig(OUT/'layerwise_probes.png',dpi=180); fig.savefig(OUT/'layerwise_probes.pdf')
 print('PROBING COMPLETE',flush=True)

if __name__=='__main__': main()
