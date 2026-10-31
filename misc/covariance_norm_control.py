"""Exploratory size-matched follow-up, fixed before its predictor evaluation."""
import time
import numpy as np
import torch
from misc.covariance_velocity import OUT,BASE,PREVIOUS,WORK,SIZES,CHECKPOINT,spline_model,edit
from misc.iterated_centroid_velocity import difference
from misc.velocity_causal import configure,save_json,load_model,continue_predict
from misc.velocity_intervention import readout,bootstrap_mean,effect_metrics
from misc.centroid_velocity import summary_item,lift
from misc.iterated_velocity import atom_npz
from misc.spline_velocity import sha


def main():
    configure();directory=OUT/'norm_control';directory.mkdir(exist_ok=True)
    save_json(directory/'protocol.json',{'source_sha256':sha(__file__),
        'reason':'Combined covariance edits were larger; evaluate a per-clip equal-norm control after observing the main results.',
        'method':'Rescale combined edit to the norm of original mean-only edit for the same clip and requested delta; no further tuning.',
        'interpretation':'Dose control, not exact target-moment transport after rescaling. Exploratory follow-up.',
        'sizes':SIZES})
    lab=np.load(BASE/'labels.npz');zero=np.load(PREVIOUS/'conditions/zero.npz');keep=zero['indices'];s=lab['speed'][keep]
    x=np.load(WORK/'centroid_velocity_v1/pooled.npy');tokens=np.load(WORK/'velocity_causal_v1/layer19.npy',mmap_mode='r')
    model=dict(np.load(OUT/'model.npz'));sm=spline_model();predictor=load_model(str(CHECKPOINT))
    readouts={k:np.load(BASE/f'{k}_position_readout.npz') for k in ['predicted','teacher','context']}
    checks={}
    for size in SIZES:
        tick=time.monotonic();delta,_=edit(x[keep],s,size,model,sm);mean=difference([sm],s,size)
        factor=np.linalg.norm(mean,axis=1)/np.maximum(np.linalg.norm(delta,axis=1),1e-20);delta*=factor[:,None]
        np.testing.assert_allclose(np.linalg.norm(delta,axis=1),np.linalg.norm(mean,axis=1),atol=1e-10)
        arrays={k:np.empty((len(keep),4,2)) for k in readouts};source=np.empty(len(keep))
        for start in range(0,len(keep),16):
            ii=keep[start:start+16];end=start+len(ii);h=torch.tensor(np.array(tokens[ii]),device='cuda')
            edited=lift(h,torch.tensor(delta[start:end],dtype=torch.float32,device='cuda'))
            source[start:end]=torch.linalg.vector_norm(h.flatten(1),dim=-1).cpu().numpy()
            with torch.inference_mode():future,context=continue_predict(predictor,edited)
            future=future.cpu().numpy();context=context.cpu().numpy()
            for k in ['predicted','teacher']:arrays[k][start:end]=readout(future,readouts[k])
            arrays['context'][start:end]=readout(context,readouts['context'])
        atom_npz(directory/f'matched_{size:+.2f}.npz',**arrays,indices=keep,edit_norm=np.sqrt(1024)*np.linalg.norm(delta,axis=1),source_norm=source,factor=factor)
        checks[str(size)]={'norm_matching_max_error':float(abs(np.linalg.norm(delta,axis=1)-np.linalg.norm(mean,axis=1)).max()),'mean_scale':float(factor.mean())}
        print(size,'seconds',time.monotonic()-tick,flush=True)
    save_json(directory/'checks.json',checks)
    times=(np.arange(8,16).reshape(4,2).mean(1)-7)/24;result={}
    for split in ['B','D']:
        mask=lab['split'][keep]==split;v=lab['velocity'][keep[mask]];result[split]={}
        for size in SIZES:
            expected=(size*v/np.linalg.norm(v,axis=1,keepdims=True))@lab['world_to_pixel'][:2]
            z=np.load(directory/f'matched_{size:+.2f}.npz');mean=np.load(PREVIOUS/'conditions'/f'spline_k001_{size:+.2f}.npz')
            row=summary_item(z,zero,mask,expected,times)
            row['paired_vs_mean']=effect_metrics((z['predicted']-mean['predicted'])[mask],expected,times)
            row['relative_edit_norm']=bootstrap_mean((z['edit_norm']/z['source_norm'])[mask]);result[split][str(size)]=row
    save_json(directory/'effects.json',result)


if __name__=='__main__':main()
