import unittest
from unittest.mock import patch
import numpy as np
import torch
from sklearn.linear_model import Ridge

from misc.spline_velocity import design, standardize, protected, ridge_map, gram_mse, delta_design, select_model


class SplineTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(193)

    def cov(self,n):
        return np.column_stack((self.rng.uniform(.8,3.4,n),self.rng.uniform(-np.pi,np.pi,n),self.rng.uniform(-.8,.8,(n,2))))

    def test_periodicity_domain_and_zero(self):
        c=self.cov(12); shifted=c.copy(); shifted[:,1]+=2*np.pi
        np.testing.assert_allclose(design(c,8,2),design(shifted,8,2),atol=1e-14)
        model={'knots':8,'direction_order':2,'scale':np.ones(240)}
        np.testing.assert_array_equal(delta_design(c,0,model),np.zeros((12,240)))
        target=c.copy(); target[:,0]+=.5
        np.testing.assert_allclose(delta_design(c,.5,model),-delta_design(target,-.5,model),atol=1e-14)
        with self.assertRaises(ValueError):
            delta_design(c,5,model)

    def test_direct_ridge_and_gram_selection(self):
        ca,cb=self.cov(45),self.cov(17)
        a,b,_,_=standardize(design(ca,8,1),design(cb,8,1))
        y=self.rng.normal(size=(62,21)); y-=y[:45].mean(0)
        a,b,yt=[torch.tensor(z,dtype=torch.float64) for z in (a,b,y)]
        gram=yt@yt.T/21
        for alpha in [.1,10,1000]:
            weights=b@ridge_map(a,alpha)
            pred=weights@yt[:45]
            reference=Ridge(alpha=alpha,fit_intercept=False,solver='cholesky').fit(a.numpy(),y[:45]).predict(b.numpy())
            np.testing.assert_allclose(pred.numpy(),reference,atol=1e-10)
            self.assertAlmostEqual(float(gram_mse(weights,gram,45)),float((pred-yt[45:]).square().mean()),places=10)
        with patch('misc.spline_velocity.KNOTS',[8]),patch('misc.spline_velocity.ORDERS',[1]),patch('misc.spline_velocity.ALPHAS',[.1,10,1000]):
            best,rows=select_model(ca,cb,gram)
        self.assertEqual(best['alpha'],min(rows,key=lambda z:z['validation_mse'])['alpha'])

    def test_projection_commutes_with_fit_and_difference(self):
        y=torch.tensor(self.rng.normal(size=(31,3,12)),dtype=torch.float64)
        q=torch.linalg.qr(torch.tensor(self.rng.normal(size=(3,12,2)),dtype=torch.float64)).Q
        p=protected(y,q)
        self.assertLess(float(torch.einsum('ntc,tcr->ntr',p,q).abs().max()),1e-13)
        torch.testing.assert_close(protected(p,q),p)
        mapping=torch.tensor(self.rng.normal(size=(7,31)),dtype=torch.float64)
        c=(mapping@p.flatten(1)).reshape(7,3,12)
        original=(mapping@y.flatten(1)).reshape(7,3,12)
        torch.testing.assert_close(c,protected(original,q))
        delta=torch.tensor(self.rng.normal(size=(5,7)),dtype=torch.float64)
        edit=torch.einsum('bd,dtc->btc',delta,c)
        self.assertLess(float(torch.einsum('ntc,tcr->ntr',edit,q).abs().max()),1e-12)


if __name__=='__main__':
    unittest.main()
