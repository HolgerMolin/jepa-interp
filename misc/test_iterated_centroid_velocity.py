import unittest
import numpy as np
import torch
from sklearn.kernel_ridge import KernelRidge
from misc.iterated_centroid_velocity import fit_sequence,contribution,cumulative,difference,centered_kernel


class ResidualSplineTests(unittest.TestCase):
    def test_telescoping_and_identity(self):
        rng=np.random.default_rng(31);sa=np.linspace(.25,4,96);sb=np.linspace(.3,3.9,48)
        w=rng.normal(size=(3,24))
        def sample(s):return np.column_stack((s,np.sin(s),s*s))@w+.2*rng.normal(size=(len(s),24))
        xa,xb=sample(sa),sample(sb)
        original_a=xa.copy();original_b=xb.copy()
        models,history,selection=fit_sequence(xa,xb,sa,sb,max_rounds=4,patience=3,pca_dim=8,knot_options=(4,8))
        ra=xa.copy();rb=xb.copy()
        for model in models:
            ca=contribution(model,sa);np.testing.assert_allclose(ca.mean(0),0,atol=1e-12)
            ra-=ca;rb-=contribution(model,sb)
        np.testing.assert_allclose(ra,xa-cumulative(models,sa),atol=1e-12)
        np.testing.assert_allclose(rb,xb-cumulative(models,sb),atol=1e-12)
        np.testing.assert_allclose(ra.mean(0),xa.mean(0),atol=1e-12)
        np.testing.assert_array_equal(xa,original_a);np.testing.assert_array_equal(xb,original_b)
        s=np.linspace(.75,3.5,10)
        np.testing.assert_array_equal(difference(models,s,0),np.zeros((10,24)))
        np.testing.assert_allclose(difference(models,s,.25),-difference(models,s+.25,-.25),atol=1e-12)
        self.assertEqual(selection['selected_prefix'],int(np.argmin([selection['initial_B_residual_mse']]+[h['B_residual_mse'] for h in history])))

    def test_centered_kernel_prediction(self):
        rng=np.random.default_rng(32);a=rng.normal(size=(23,8));b=rng.normal(size=(11,8));y=rng.normal(size=23);y-=y.mean()
        k=np.exp(-np.square(a[:,None]-a[None]).sum(-1)/8)
        kt=np.exp(-np.square(b[:,None]-a[None]).sum(-1)/8)
        kc,bc=centered_kernel(torch.tensor(k),torch.tensor(kt))
        np.testing.assert_allclose(kc.numpy().mean(0),0,atol=1e-14)
        alpha=.1
        ref=KernelRidge(alpha=alpha,kernel='precomputed').fit(kc.numpy(),y).predict(bc.numpy())
        own=(bc@torch.linalg.solve(kc+alpha*torch.eye(len(a)),torch.tensor(y))).numpy()
        np.testing.assert_allclose(own,ref,atol=1e-7)


if __name__=='__main__':unittest.main()
