import hashlib, json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import av
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'
OUT.mkdir(exist_ok=True)
SEED=20260929

def inspect(item):
    task, row, base=item
    p=base/row['video']; m=json.loads((base/row['metadata']).read_text())
    with av.open(str(p)) as c:
        frames=list(c.decode(video=0))
        assert len(frames)==16 and all(f.width==256 and f.height==256 for f in frames), p
        assert float(c.streams.video[0].average_rate)==24, p
    return dict(task=task,id=row['id'],video=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),metadata=m)

items=[]
for task in ['direction','speed','acceleration']:
    base=ROOT/'vjepa-physics-takehome-4E00/data'/task
    rows=[json.loads(s) for s in (base/'manifest.jsonl').read_text().splitlines()]
    assert len({r['id'] for r in rows})==len(rows)
    items.extend((task,r,base) for r in rows)
with ThreadPoolExecutor(8) as pool: records=list(pool.map(inspect,items))
# A duplicate requires a grouped split, not the ordinary stratified split below.
assert len({r['sha256'] for r in records})==len(records), 'Duplicate video hashes: implement grouped splitting before proceeding'
rng=np.random.default_rng(SEED)
summary={}
for task in ['direction','speed','acceleration']:
    rr=[r for r in records if r['task']==task]
    key={'direction':'theta_degrees','speed':'speed_mps','acceleration':'acceleration_mps2'}[task]
    labels=np.array([r['metadata'][key] for r in rr])
    for label in np.unique(labels):
        idx=np.flatnonzero(labels==label); rng.shuffle(idx)
        n=len(idx); a=round(.50*n); b=round(.15*n); c=round(.15*n)
        for split,ii in zip('ABCD',np.split(idx,[a,a+b,a+b+c])):
            for i in ii: rr[i]['split']=split
    nuisance=np.array([[*r['metadata']['start_position_xy_m'],np.sin(np.deg2rad(r['metadata']['theta_degrees'])),np.cos(np.deg2rad(r['metadata']['theta_degrees']))] for r in rr])
    summary[task]={'n':len(rr),'label_counts':dict(zip(*[x.tolist() for x in np.unique(labels,return_counts=True)])),
        'splits':{s:sum(r['split']==s for r in rr) for s in 'ABCD'},
        'label_nuisance_pearson_xy_sin_cos':np.corrcoef(np.column_stack([labels,nuisance]).T)[0,1:].tolist(),
        'motion_counts':{v:sum(r['metadata']['motion']==v for r in rr) for v in {r['metadata']['motion'] for r in rr}},
        'split_summary':{s:{'label_mean':float(np.mean([r['metadata'][key] for r in rr if r['split']==s])), 'position_mean':np.mean([r['metadata']['start_position_xy_m'] for r in rr if r['split']==s],axis=0).tolist()} for s in 'ABCD'}}
(OUT/'splits.json').write_text(json.dumps({'seed':SEED,'records':records},indent=2)+'\n')
(OUT/'audit.json').write_text(json.dumps({'unique_hashes':len(records),'all_videos_decoded':True,'datasets':summary},indent=2)+'\n')
print(json.dumps(summary,indent=2),flush=True)
