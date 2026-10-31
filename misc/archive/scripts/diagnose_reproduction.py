"""Post-hoc diagnostics; preserve primary run and do not tune it on these results."""
import json, hashlib
from pathlib import Path
import numpy as np, av, torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from transformers import VJEPA2Model
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/diagnostics'; OUT.mkdir(exist_ok=True)
SCRATCH=Path('/scratch/users/hmolin/jepa-interp'); records=json.loads((ROOT/'results/splits.json').read_text())['records']
results={'status':'post-hoc diagnostic on previously viewed splits; not new confirmatory evidence'}

def metric(y,p):
 a=np.arctan2(y[:,0],y[:,1]); b=np.arctan2(p[:,0],p[:,1])
 return {'r2':float(r2_score(y,p)),'mae_degrees':float(np.mean(np.abs((b-a+np.pi)%(2*np.pi)-np.pi))*180/np.pi)}

def cached(task):
 rr=[r for r in records if r['task']==task]; features=[]; ids=[]
 for p in sorted((SCRATCH/'features').glob(task+'_[0-9]*.npz')):
  with np.load(p) as f: features.append(f['features']); ids.extend(f['ids'].tolist())
 assert ids==[r['id'] for r in rr]
 return rr,np.concatenate(features)

# Independent primal sklearn solve on real data at two layers in every task.
results['independent_solver']={}
for task in ['direction','speed','acceleration']:
 rr,x=cached(task); a=np.array([r['split']=='A' for r in rr]); d=np.array([r['split']=='D' for r in rr])
 if task=='direction':
  angles=np.deg2rad([r['metadata']['theta_degrees'] for r in rr]); y=np.stack([np.sin(angles),np.cos(angles)],1)
 else: y=np.array([r['metadata']['speed_mps' if task=='speed' else 'acceleration_mps2'] for r in rr])[:,None]
 results['independent_solver'][task]={}
 for layer in [1,19]:
  with np.load(ROOT/f'results/probe_{task}_layer{layer:02d}.npz') as f:
   ref=Ridge(alpha=float(f['alpha']),solver='svd').fit(x[a,layer-1].astype(float),y[a])
   pred=ref.predict(x[d,layer-1].astype(float)).reshape(y[d].shape); error=float(abs(pred-f['test_prediction']).max())
   np.testing.assert_allclose(pred,f['test_prediction'],atol=1e-6,rtol=1e-5)
   results['independent_solver'][task][str(layer)]={'max_prediction_difference':error,'r2':float(r2_score(y[d],pred))}
 print('Independent solver verified',task,flush=True)

rr,x=cached('direction'); angles=np.deg2rad([r['metadata']['theta_degrees'] for r in rr]); y=np.stack([np.sin(angles),np.cos(angles)],1)
a=np.array([r['split']=='A' for r in rr]); b=np.array([r['split']=='B' for r in rr]); d=np.array([r['split']=='D' for r in rr]);
# Hold optimizer setting fixed to previously selected values; permute labels afresh many times.
results['permutation_null']={}
for layer in [1,19]:
 with np.load(ROOT/f'results/probe_direction_layer{layer:02d}.npz') as f: alpha=float(f['alpha'])
 z=x[a,layer-1].astype(float); xm=z.mean(0); z-=xm; u,s,vt=np.linalg.svd(z,full_matrices=False)
 predictions=[]; rng=np.random.default_rng(8891)
 for _ in range(30):
  yp=y[a][rng.permutation(sum(a))]; ym=yp.mean(0)
  w=(vt.T*(s/(s*s+alpha)))@(u.T@(yp-ym)); pp=(x[d,layer-1]-xm)@w+ym
  predictions.append(metric(y[d],pp))
 results['permutation_null'][str(layer)]={'alpha_fixed':alpha,'repeats':30,'mean_r2':float(np.mean([p['r2'] for p in predictions])),'mean_mae':float(np.mean([p['mae_degrees'] for p in predictions])),'all':predictions}

# Reverse regime transfer uses only A/B for fitting/selection, scores relevant D examples.
results['direction_regime_transfer']={}
regime=np.array([r['metadata']['motion'] for r in rr]); alphas=[1e-4,.001,.01,.1,1,10,100,1000,10000]
for src,dst in [('velocity','acceleration'),('acceleration','velocity')]:
 results['direction_regime_transfer'][src+'->'+dst]={}
 for layer in [1,19]:
  aa=a&(regime==src); bb=b&(regime==src); dd=d&(regime==dst)
  best=None
  for alpha in alphas:
   mod=Ridge(alpha=alpha).fit(x[aa,layer-1].astype(float),y[aa]); score=metric(y[bb],mod.predict(x[bb,layer-1]))['mae_degrees']
   if best is None or score<best[0]: best=(score,mod,alpha)
  results['direction_regime_transfer'][src+'->'+dst][str(layer)]={'alpha':best[2],'test_n':int(dd.sum()),**metric(y[dd],best[1].predict(x[dd,layer-1]))}

# Pixel-to-label alignment: disk centroid displacement, independent of the encoder.
def decode(r):
 with av.open(r['video']) as c: return np.stack([f.to_ndarray(format='rgb24') for f in c.decode(video=0)])
selected=[]
for task in ['direction','speed','acceleration']:
 tr=[r for r in records if r['task']==task]
 selected.extend(tr[i] for i in np.linspace(0,len(tr)-1,12,dtype=int))
pixel=[]
for r in selected:
 frames=decode(r); cent=[]
 for frame in [frames[0],frames[-1]]:
  mask=(frame.max(axis=2).astype(float)-frame.min(axis=2).astype(float)>40)&(frame.max(axis=2)>80)
  yy,xx=np.nonzero(mask)
  cent.append([xx.mean(),yy.mean()] if len(xx) else None)
 if any(c is None for c in cent):
  pixel.append({'task':r['task'],'id':r['id'],'missing_visible_disk_at_endpoint':True}); continue
 delta=np.subtract(cent[1],cent[0]); theta=float(np.rad2deg(np.arctan2(-delta[1],delta[0]))%360)
 err=abs((theta-r['metadata']['theta_degrees']+180)%360-180)
 pixel.append({'task':r['task'],'id':r['id'],'metadata_angle':r['metadata']['theta_degrees'],'pixel_angle_y_up':theta,'error':err})
results['pixel_label_check']=pixel
(OUT/'audit_partial.json').write_text(json.dumps(results,indent=2)+'\n')

# Fresh encoder extraction independently validates all hooks through hidden-state API.
torch.set_num_threads(4); torch.manual_seed(123)
model=VJEPA2Model.from_pretrained(str(SCRATCH/'checkpoints/vjepa2-vitl-fpc64-256'),local_files_only=True,attn_implementation='sdpa').eval().cuda(); model.requires_grad_(False)
mean=torch.tensor([.485,.456,.406],device='cuda')[None,None,:,None,None]; std=torch.tensor([.229,.224,.225],device='cuda')[None,None,:,None,None]

def prep(batch,static=False):
 frames=[decode(r) for r in batch]
 if static: frames=[np.repeat(f[:1],16,axis=0) for f in frames]
 return (torch.from_numpy(np.stack(frames)).permute(0,1,4,2,3).cuda().float()/255-mean)/std

results['fresh_hidden_state_checks']=[]
with torch.inference_mode():
 for task in ['direction','speed','acceleration']:
  tr,tx=cached(task); ix=[3,len(tr)//2,len(tr)-2]
  pp=prep([tr[i] for i in ix]); out=model(pp,skip_predictor=True,output_hidden_states=True)
  # hidden_states[0] is patch embeddings, [1:24] block 1..23, [24] normalized block 24.
  for layer in [1,8,19,23]:
   fresh=out.hidden_states[layer].mean(1).cpu().numpy()
   np.testing.assert_allclose(fresh,tx[ix,layer-1],atol=2e-5,rtol=2e-5)
   results['fresh_hidden_state_checks'].append({'task':task,'layer':layer,'max_error':float(abs(fresh-tx[ix,layer-1]).max())})
 print('Fresh extraction and hidden-state API verified',flush=True)
 # Train a true static-input baseline, plus an embedding-only baseline before any transformer.
 emb=[]; static=[]; normalized=[]
 for start in range(0,len(rr),4):
  batch=rr[start:start+4]; pp=prep(batch)
  emb.append(model.encoder.embeddings(pp).mean(1).cpu().numpy())
  original=model(pp,skip_predictor=True,output_hidden_states=True)
  normalized.append(torch.stack([model.encoder.layernorm(original.hidden_states[k]).mean(1) for k in [1,8,19]],1).cpu().numpy())
  del original
  ss=prep(batch,static=True); out=model(ss,skip_predictor=True,output_hidden_states=True)
  static.append(torch.stack([out.hidden_states[1].mean(1),out.hidden_states[19].mean(1)],1).cpu().numpy())
  if start%100==0: print('Static and patch-only diagnostics',start,len(rr),flush=True)
 np.savez(SCRATCH/'diagnostic_direction_features.npz',patch=np.concatenate(emb),static=np.concatenate(static),normalized=np.concatenate(normalized),ids=[r['id'] for r in rr])

results['refitted_controls']={}
for name,z in [('patch_embedding_only',np.concatenate(emb)),('static_block1',np.concatenate(static)[:,0]),('static_block19',np.concatenate(static)[:,1]),('normalized_block1',np.concatenate(normalized)[:,0]),('normalized_block8',np.concatenate(normalized)[:,1]),('normalized_block19',np.concatenate(normalized)[:,2])]:
 best=None
 for alpha in alphas:
  mod=Ridge(alpha=alpha,solver='svd').fit(z[a].astype(float),y[a]); score=metric(y[b],mod.predict(z[b]))['mae_degrees']
  if best is None or score<best[0]: best=(score,mod,alpha)
 results['refitted_controls'][name]={'alpha':best[2],**metric(y[d],best[1].predict(z[d]))}
(OUT/'audit_results.json').write_text(json.dumps(results,indent=2)+'\n')
print('DIAGNOSTICS COMPLETE',flush=True)
