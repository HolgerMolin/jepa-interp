"""Audit readability after the selected conditional whitening transformation."""
import json
import numpy as np
from covariance_velocity import OUT,BASE,WORK,spline_model,conditional_covariance,matrix_power
from iterated_centroid_velocity import contribution,kernel_candidates,kernel_predict
from diagnose_spline_residuals import scores
from velocity_causal import configure,save_json
from iterated_velocity import atom_npz
from spline_velocity import sha


def main():
    configure();lab=np.load(BASE/'labels.npz');s=lab['speed'];x=np.load(WORK/'centroid_velocity_v1/pooled.npy')
    a,b,d=[np.flatnonzero(lab['split']==p) for p in ['A','B','D']]
    m=dict(np.load(OUT/'model.npz'));dim=int(m['dim']);basis=m['components'][:dim];scale=m['scale'][:dim]
    r=x-m['mean']-contribution(spline_model(),s)-m['residual_pca_mean'];z=(r@basis.T)/scale
    cov=conditional_covariance(m['za'],m['sa'],s,float(m['bandwidth']),float(m['shrinkage']))
    white=np.einsum('ni,nij->nj',z,matrix_power(cov,-.5))
    variants={'raw_residual_coordinates':z,'whitened_coordinates':white,
              'whitened_full_residual':r+((white-z)*scale)@basis}
    settings={};rows={};result={};predictions={}
    for name,f in variants.items():
        settings[name],rows[name]=kernel_candidates(f[a],f[b],s[a],s[b])
    save_json(OUT/'whitening_audit_selection.json',{'source_sha256':sha(__file__),'settings':settings,'candidates':rows,
              'caution':'Whitening, like residual subtraction, uses known speed; this audit cannot certify label-free independence.'})
    for name,f in variants.items():
        result[name]={}
        for kind,setting in settings[name].items():
            pp=kernel_predict(f[a],f[np.r_[b,d]],s[a],setting);predictions[name+'_'+kind]=pp
            result[name][kind]={'B':scores(pp[:len(b)],s[b]),'D':scores(pp[len(b):],s[d])}
    save_json(OUT/'whitening_audit.json',result);atom_npz(OUT/'whitening_predictions.npz',**predictions,indices=np.r_[b,d])
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
