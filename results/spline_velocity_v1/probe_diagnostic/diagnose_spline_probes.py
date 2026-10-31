"""Read the completed spline interventions with the frozen iterated velocity probes.

No models are fitted and no predictor outcomes select probes. Primary summaries
give every token equal weight, averaging its B-eligible probes first. Uncertainty
resamples clips, not correlated token/probe observations.
"""
import json
from pathlib import Path
import shutil
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from spline_velocity import ROOT, SIZES, covariates, delta_design, protected, sha
from velocity_causal import configure, save_json
from velocity_intervention import bootstrap_mean
from iterated_velocity import atom_npz, CAPS

BASE=ROOT/'results/velocity_causal_v1'
SPLINE=ROOT/'results/spline_velocity_v1'
OUT=SPLINE/'probe_diagnostic'
WORK=Path('/scratch/users/hmolin/jepa-interp')


def token_average(values,tokens):
    """values [probe,clip,...]; equal weight within each token then over tokens."""
    unique,counts=np.unique(tokens,return_counts=True)
    weights=np.zeros(len(values))
    for token,count in zip(unique,counts):
        weights[tokens==token]=1/(len(unique)*count)
    return np.einsum('p,pn...->n...',weights,values)


def metrics(change,expected,tokens):
    norm=np.linalg.norm(expected,axis=-1)
    unit=expected/norm[:,None]
    gain=np.einsum('pnd,nd->pn',change,unit)/norm[None]
    perp=np.abs(change[:,:,0]*unit[None,:,1]-change[:,:,1]*unit[None,:,0])/norm[None]
    error=np.linalg.norm(change-expected[None],axis=-1)/norm[None]
    perclip=token_average(gain,tokens)
    return {'probes':len(tokens),'tokens':len(np.unique(tokens)),
            'aligned_change_gain':bootstrap_mean(perclip),
            'relative_vector_error':bootstrap_mean(token_average(error,tokens)),
            'relative_perpendicular_change':bootstrap_mean(token_average(perp,tokens)),
            'fraction_correct_sign':bootstrap_mean(token_average((gain>0).astype(float),tokens)),
            'fraction_within_25pct_vector_error':bootstrap_mean(token_average((error<=.25).astype(float),tokens)),
            'per_clip_mean_gain_quantiles':np.quantile(perclip,[0,.1,.25,.5,.75,.9,1]).tolist()}


def main():
    configure(); OUT.mkdir(exist_ok=True)
    labels=np.load(BASE/'labels.npz')
    zero=np.load(SPLINE/'conditions/zero.npz'); keep=zero['indices']; n=len(keep)
    model=np.load(SPLINE/'model.npz')
    files=sorted((WORK/'iterated_velocity_v1/chunks').glob('tokens_*.npz'))
    counts=np.load(ROOT/'results/iterated_velocity_v1/token_counts.npz')['eligible_counts']
    total=int(counts.sum())
    protocol={'question':'Do frozen B-eligible iterated velocity probes report the intended changes after the completed spline edits?',
              'source_sha256':sha(__file__),'spline_model_sha256':sha(SPLINE/'model.npz'),
              'spline_coefficients_sha256':sha(WORK/'spline_velocity_v1/coefficients.npy'),
              'probe_chunks_sha256':{p.name:sha(p) for p in files},
              'evaluation_ids':labels['ids'][keep].tolist(),'eligible_probes':total,
              'selection':'unchanged B eligibility: BOTH vx/vy R2 >= .5; report first, subsequent, and all eligible probes',
              'quantity':'probe(H + edit) - probe(H), including float32 token-addition rounding; target delta_v',
              'aggregation':'average probes within token, then tokens equally, then clips; CIs bootstrap clips only',
              'interpretation':'diagnostic using construction probes; linear response is exact by design, not independent validation; D historically inspected'}
    save_json(OUT/'protocol.json',protocol)
    changes=np.empty((4,total,n,2),dtype=np.float32)
    clean=np.empty((total,n,2),dtype=np.float32)
    tokens=np.empty(total,dtype=int); iterations=np.empty(total,dtype=int); orders=np.empty(total,dtype=int)
    validation=np.empty((total,2))
    coeff=np.load(WORK/'spline_velocity_v1/coefficients.npy',mmap_mode='r')
    x=np.load(WORK/'velocity_causal_v1/layer19.npy',mmap_mode='r')
    operators=np.load(WORK/'iterated_velocity_v1/operators.npy',mmap_mode='r')
    designs=[torch.tensor(delta_design(covariates(labels,keep),size,model),dtype=torch.float32,device='cuda') for size in SIZES]
    squared_norm=np.zeros((4,n)); max_linear_error=0.; max_direct_error=0.
    offset=0; started=time.monotonic()
    for path in files:
        z=np.load(path); ii=z['token_indices']; t=len(ii)
        # Map orthogonal residual probes back to the original token coordinates:
        # the saved effective weights already include all prior projections.
        rr,tt=np.nonzero(z['eligible'])
        weights=torch.tensor(z['coef'][rr,tt],dtype=torch.float64,device='cuda')
        local=torch.tensor(tt,device='cuda')
        count=len(rr); sl=slice(offset,offset+count)
        tokens[sl]=ii[tt]; iterations[sl]=rr
        for k,(r,j) in enumerate(zip(rr,tt)):
            orders[offset+k]=int(z['eligible'][:r,j].sum())+1
        validation[sl]=z['validation_r2'][rr,tt]
        q=torch.tensor(z['position_basis'],dtype=torch.float32,device='cuda')
        c=torch.tensor(np.array(coeff[:,ii]),dtype=torch.float32,device='cuda')
        op=torch.tensor(np.array(operators[CAPS.index(32),ii]),dtype=torch.float64,device='cuda')
        residual=torch.einsum('poc,pcd->pod',weights,op[local])-torch.eye(2,dtype=torch.float64,device='cuda')
        max_linear_error=max(max_linear_error,float(residual.abs().max()))
        for start in range(0,n,16):
            end=min(start+16,n)
            h=torch.tensor(np.array(x[keep[start:end]][:,ii]),dtype=torch.float32,device='cuda')
            mean=torch.tensor(z['mean'],dtype=torch.float64,device='cuda')
            ym=torch.tensor(z['y_mean'],dtype=torch.float64,device='cuda')
            before=torch.einsum('bpc,poc->pbo',(h.double()-mean)[:,local],weights)+ym
            clean[sl,start:end]=before.cpu().numpy()
            for si,size in enumerate(SIZES):
                edit=protected(torch.einsum('bd,dtc->btc',designs[si][start:end],c),q)
                after_tokens=h+edit
                actual_edit=after_tokens.double()-h.double()
                result=torch.einsum('bpc,poc->pbo',actual_edit[:,local],weights)
                changes[si,sl,start:end]=result.cpu().numpy()
                squared_norm[si,start:end]+=edit.double().square().sum((1,2)).cpu().numpy()
                if start==0:
                    direct=torch.einsum('bpc,poc->pbo',(after_tokens.double()-mean)[:,local],weights)+ym-before
                    max_direct_error=max(max_direct_error,float((direct-result).abs().max()))
        offset+=count
        if int(ii[-1]+1)%128==0:
            print(f'probe responses through token {ii[-1]+1}/1024; {(time.monotonic()-started)/60:.1f}m',flush=True)
    assert offset==total
    if max_linear_error>1e-6 or max_direct_error>1e-10:
        raise AssertionError((max_linear_error,max_direct_error))
    norm_error=0.
    for si,size in enumerate(SIZES):
        original=np.load(SPLINE/'conditions'/f'spline_{size:+.2f}.npz')['edit_norm']
        reconstructed=np.sqrt(squared_norm[si])
        np.testing.assert_allclose(reconstructed,original,rtol=3e-6,atol=1e-4)
        norm_error=max(norm_error,float(np.max(np.abs(reconstructed-original)/original)))
    atom_npz(OUT/'responses.npz',delta_speed=np.array(SIZES),change=changes,clean_prediction=clean,
             token=tokens,iteration=iterations,eligible_order=orders,validation_r2=validation,
             indices=keep,ids=labels['ids'][keep],split=labels['split'][keep])
    summary={'splits':{},'checks':{'linear_operator_max_constraint_error':max_linear_error,
              'direct_before_after_max_error':max_direct_error,'reconstructed_norm_max_relative_error':norm_error,
              'seconds':time.monotonic()-started}}
    groups={'first':orders==1,'subsequent':orders>1,'all':np.ones(total,dtype=bool)}
    for part in ['B','D']:
        mask=labels['split'][keep]==part
        v=labels['velocity'][keep[mask]]
        clean_part=clean[:,mask]
        r2=1-((clean_part-v[None])**2).sum(1)/((v-v.mean(0))**2).sum(0)
        report={'clean_r2_quantiles':np.quantile(r2,[.1,.5,.9],axis=0).tolist(),'conditions':{}}
        for si,size in enumerate(SIZES):
            expected=size*v/np.linalg.norm(v,axis=1,keepdims=True)
            change=changes[si][:,mask]
            rows={name:metrics(change[select],expected,tokens[select]) for name,select in groups.items()}
            # Within-token disagreement among reliable sequential readouts.
            unit=expected/np.linalg.norm(expected,axis=1,keepdims=True)
            gain=np.einsum('pnd,nd->pn',change,unit)/abs(size)
            dispersions=[]
            means=np.empty((1024,len(v)))
            for token in range(1024):
                select=tokens==token
                means[token]=gain[select].mean(0)
                if select.sum()>1:
                    dispersions.append(gain[select].std(0))
            rows['within_token_probe_gain_std']=bootstrap_mean(np.mean(dispersions,axis=0))
            rows['token_mean_gain_quantiles']=np.quantile(means.mean(1),[0,.1,.25,.5,.75,.9,1]).tolist()
            # Group by eligible order without treating low-quality residual probes as evidence.
            rows['by_eligible_order']={str(k):metrics(change[orders==k],expected,tokens[orders==k])
                                       for k in range(1,int(orders.max())+1)}
            report['conditions'][f'{size:+.2f}']=rows
        summary['splits'][part]=report
    save_json(OUT/'summary.json',summary)
    write_report(summary,changes,tokens,orders,labels,keep)
    shutil.copy2(__file__,OUT/Path(__file__).name)
    print(OUT/'REPORT.md',flush=True)


def write_report(summary,changes,tokens,orders,labels,keep):
    d=summary['splits']['D']['conditions']
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for name,color in [('first','#1976b5'),('subsequent','#ce7320'),('all','#276d45')]:
        vals=[d[f'{s:+.2f}'][name]['aligned_change_gain'] for s in SIZES]
        means=np.array([v['mean'] for v in vals]); lo=np.array([v['ci95'][0] for v in vals]); hi=np.array([v['ci95'][1] for v in vals])
        axes[0].errorbar(SIZES,means,yerr=np.maximum([means-lo,hi-means],0),fmt='o-',color=color,label=name,capsize=3)
    axes[0].axhline(1,color='black',linestyle='--',label='Requested change / linear construction')
    axes[0].axhline(0,color='gray',linewidth=.6)
    axes[0].set(xlabel='Requested speed change (m/s)',ylabel='Probe-reported change / requested change',title='Layer-19 velocity probe response')
    axes[0].legend(fontsize=8); axes[0].grid(alpha=.2)
    for size,color in [(-.5,'#1976b5'),(.5,'#ce7320')]:
        rows=d[f'{size:+.2f}']['by_eligible_order']
        kk=[int(k) for k,v in rows.items() if v['tokens']>=20]
        axes[1].plot(kk,[rows[str(k)]['aligned_change_gain']['mean'] for k in kk],'o-',label=f'{size:+.2f} m/s',color=color)
    axes[1].axhline(1,color='black',linestyle='--'); axes[1].axhline(0,color='gray',linewidth=.6)
    axes[1].set(xlabel='Order among B-eligible probes (≥20 tokens)',ylabel='Mean probe change gain',title='Response across redundant readouts')
    axes[1].legend(); axes[1].grid(alpha=.2)
    fig.savefig(OUT/'probe_response.png',dpi=170); plt.close(fig)
    lines=['# Velocity-probe diagnostic of spline edits','',
           'This diagnostic reads the completed spline edits with the existing, frozen, B-eligible iterated velocity probes. '
           'It fits no new model and does not change spline settings or predictor outcomes. '
           'The primary quantity is `probe(H + edit) − probe(H)`, projected onto the requested velocity change and divided by its magnitude. '
           'A gain of 1 means the probe reports the requested change; 0 means no aligned change; negative means the opposite direction.','',
           f'There are **{len(tokens)} eligible two-output probes over 1,024 tokens**, with {int(sum(labels["split"][keep]=="D"))} D clips. '
           'Each token gets equal weight, with its probes averaged first. First and subsequent probes are reported separately. '
           'Intervals bootstrap clips only; token/probe observations are not treated as independent samples.','',
           '| Requested Δspeed | First-probe gain [95% CI] | Subsequent-probe gain [95% CI] | All-probe gain [95% CI] | Correct-sign fraction | Within 25% vector-error fraction |',
           '|---:|---|---|---|---:|---:|']
    def fmt(v):
        return f'{v["mean"]:.3f} [{v["ci95"][0]:.3f}, {v["ci95"][1]:.3f}]'
    for size in SIZES:
        row=d[f'{size:+.2f}']
        lines.append(f'| {size:+.2f} | '+ ' | '.join(fmt(row[g]['aligned_change_gain']) for g in ['first','subsequent','all'])+
                     f' | {row["all"]["fraction_correct_sign"]["mean"]:.3f} | {row["all"]["fraction_within_25pct_vector_error"]["mean"]:.3f} |')
    lines+=['','## Consistency and off-target changes','',
            '| Δspeed | Mean relative vector error | Mean relative perpendicular change | Mean within-token probe-gain SD |',
            '|---:|---:|---:|---:|']
    for size in SIZES:
        row=d[f'{size:+.2f}']
        lines.append(f'| {size:+.2f} | {row["all"]["relative_vector_error"]["mean"]:.3f} | '
                     f'{row["all"]["relative_perpendicular_change"]["mean"]:.3f} | {row["within_token_probe_gain_std"]["mean"]:.3f} |')
    lines+=['','Relative vector error is `||decoded Δv − requested Δv|| / ||requested Δv||`; '
            'perpendicular change is normalized by the same requested magnitude. '
            'Within-token SD is computed across eligible probes, then averaged over tokens with at least two probes and over clips. '
            'It measures disagreement of redundant readouts, not uncertainty in the mean. '
            'The per-order plot changes token membership as order increases; it is descriptive rather than a matched-token depth comparison.','',
            '## Checks and interpretation','',
            'The saved effective probe coefficients already include the sequential null-space projections, so they can be applied '
            'directly to full layer-19 tokens after subtracting the saved training mean. '
            'Predictions include the saved physical-unit intercept; it cancels in before/after differences. '
            'The reconstructed edit uses the frozen spline coefficients, identical position projection, and float32 token addition.','',
            f'- Linear all-eligible operator response error versus the exact requested change: {summary["checks"]["linear_operator_max_constraint_error"]:.3e}.',
            f'- Direct before/after prediction versus coefficient-times-actual-edit error: {summary["checks"]["direct_before_after_max_error"]:.3e}.',
            f'- Reconstructed spline edit-norm maximum relative difference versus the completed experiment: {summary["checks"]["reconstructed_norm_max_relative_error"]:.3e}.','',
            'The linear method satisfies these readouts by construction; its gain of 1 is a numerical check, not independent evidence of causality. '
            'Spline results here diagnose agreement with those same velocity representations. '
            'Failure to shift them does not exclude other velocity representations, and agreement would not guarantee natural edits. '
            'D has already been inspected; no settings are selected from this diagnostic.','',
            '- [Probe-response plot](probe_response.png).',
            '- `responses.npz`: baseline predictions and every eligible probe’s per-clip velocity change.',
            '- `summary.json`: B/D aggregates, per-order results, unedited R² diagnostics, and numerical checks.',
            '- `protocol.json`: frozen inputs, selected probes, hashes, and aggregation rules.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))


if __name__=='__main__':
    main()
