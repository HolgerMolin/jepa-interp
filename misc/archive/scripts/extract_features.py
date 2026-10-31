import json, time, importlib.metadata, os
from pathlib import Path
import av, numpy as np, torch
from transformers import VJEPA2Model
ROOT=Path(__file__).resolve().parents[1]
SCRATCH=Path('/scratch/users/hmolin/jepa-interp')
OUT=SCRATCH/'features'; OUT.mkdir(exist_ok=True)
records=json.loads((ROOT/'results/splits.json').read_text())['records']
torch.manual_seed(20260929); torch.set_num_threads(4)
assert torch.cuda.is_available()
model=VJEPA2Model.from_pretrained(str(SCRATCH/'checkpoints/vjepa2-vitl-fpc64-256'),local_files_only=True,attn_implementation='sdpa').eval().cuda()
model.requires_grad_(False)
mean=torch.tensor([.485,.456,.406],device='cuda')[None,None,:,None,None]
std=torch.tensor([.229,.224,.225],device='cuda')[None,None,:,None,None]
pooled={}; last={}
def hook(i):
 def capture(module,inputs,outputs):
    z=outputs[0] if isinstance(outputs,tuple) else outputs
    assert z.shape[1:]==(2048,1024), z.shape
    pooled[i]=z.float().mean(1)
    if i==23: last['z']=z
 return capture
handles=[b.register_forward_hook(hook(i)) for i,b in enumerate(model.encoder.layer)]

def load(r,control=None):
 with av.open(r['video']) as c: a=np.stack([f.to_ndarray(format='rgb24') for f in c.decode(video=0)])
 if control=='static': a=np.repeat(a[:1],16,axis=0)
 if control=='shuffle': a=a[np.random.default_rng(20260929+r['id']).permutation(16)]
 return torch.from_numpy(a.copy()).permute(0,3,1,2)

@torch.inference_mode()
def forward(rr,control=None):
 x=torch.stack([load(r,control) for r in rr]).cuda().float()/255
 x=(x-mean)/std
 result=model.get_vision_features(x)
 normalized=model.encoder.layernorm(last['z'])
 torch.testing.assert_close(result,normalized,rtol=1e-5,atol=1e-6)
 y=torch.stack([pooled[i] for i in range(24)],dim=1).cpu().numpy()
 assert np.isfinite(y).all()
 return y

t0=time.time(); first=forward(records[:2]); second=forward(records[:2])
np.testing.assert_allclose(first,second,rtol=1e-5,atol=1e-6)
smoke={'device':torch.cuda.get_device_name(),'input_frames':16,'tokens':2048,'layers':24,'features':1024,'dtype':'float32','batch_size':2,'seconds_two_forwards':time.time()-t0,'hook_final_norm_agreement':True,'repeat_max_abs':float(abs(first-second).max()),'preprocessing':'RGB; native 256x256; no resize/crop; divide by 255; checkpoint ImageNet mean/std','versions':{p:importlib.metadata.version(p) for p in ['torch','torchvision','transformers','av','numpy']}}
(ROOT/'results/model_smoke.json').write_text(json.dumps(smoke,indent=2)+'\n'); print(smoke,flush=True)
for task in ['direction','speed','acceleration']:
 rr=[r for r in records if r['task']==task]; begin=time.time()
 # Atomic shards make a stopped job resumable without treating partial arrays as complete.
 for start in range(0,len(rr),32):
    path=OUT/f'{task}_{start:04d}.npz'; chunk=rr[start:start+32]
    if path.exists():
      with np.load(path) as f:
        assert f['features'].shape==(len(chunk),24,1024) and np.isfinite(f['features']).all()
        assert list(f['ids'])==[r['id'] for r in chunk]
      continue
    y=np.concatenate([forward(chunk[i:i+2]) for i in range(0,len(chunk),2)])
    temp=path.with_suffix('.tmp.npz'); np.savez(temp,features=y,ids=[r['id'] for r in chunk]); temp.replace(path)
    print(task,start+len(chunk),len(rr),'seconds',round(time.time()-begin,1),flush=True)
 # Fixed, label-stratified D subset: one example per primary target level.
 key={'direction':'theta_degrees','speed':'speed_mps','acceleration':'acceleration_mps2'}[task]
 selected={}
 for r in rr:
    if r['split']=='D': selected.setdefault(r['metadata'][key],r)
 cc=list(selected.values())
 for control in ['static','shuffle']:
    path=OUT/f'{task}_control_{control}.npz'
    if not path.exists():
      y=np.concatenate([forward(cc[i:i+2],control) for i in range(0,len(cc),2)])
      temp=path.with_suffix('.tmp.npz'); np.savez(temp,features=y,ids=[r['id'] for r in cc]); temp.replace(path)
print('EXTRACTION COMPLETE',flush=True)
