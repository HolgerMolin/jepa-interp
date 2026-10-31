import unittest
import numpy as np
from covariance_velocity import conditional_covariance,matrix_power,transport,conditional_nll
from diagnose_spline_residuals import curve_span,remove_span


class CovarianceTests(unittest.TestCase):
    def test_transport_covariance_and_inverse(self):
        rng=np.random.default_rng(41);a=rng.normal(size=(4,6,6));b=rng.normal(size=(4,6,6))
        cs=a@a.swapaxes(-1,-2)+np.eye(6);ct=b@b.swapaxes(-1,-2)+np.eye(6)
        transform=matrix_power(cs,-.5)@matrix_power(ct,.5)
        np.testing.assert_allclose(transform.swapaxes(-1,-2)@cs@transform,ct,atol=1e-12)
        z=rng.normal(size=(4,6));out=transport(z,cs,ct)
        np.testing.assert_allclose(transport(out,ct,cs),z,atol=1e-12)
        np.testing.assert_allclose(transport(z,cs,cs),z,atol=1e-12)

    def test_shrinkage_and_comparable_likelihood(self):
        rng=np.random.default_rng(42);za=rng.normal(size=(100,8));sa=np.linspace(.25,4,100);q=np.linspace(.5,3.5,11)
        z=rng.normal(size=(11,8))
        for dim in [2,4,8]:
            cov=conditional_covariance(za[:,:dim],sa,q,.35,1.)
            np.testing.assert_allclose(conditional_nll(z,cov),.5*np.square(z).sum(1),atol=1e-12)
            regularized=conditional_covariance(za[:,:dim],sa,q,.35,.25)
            self.assertGreaterEqual(np.linalg.eigvalsh(regularized).min(),.25-1e-12)

    def test_label_independent_span_projection(self):
        rng=np.random.default_rng(43);basis=np.linalg.qr(rng.normal(size=(20,6)))[0].T
        centroids=rng.normal(size=(4,6));center=centroids.mean(0)
        model={'centroids':centroids,'curve_mean':center,'components':basis}
        q=curve_span(model);x=rng.normal(size=(7,20));mean=rng.normal(size=20)
        shift=rng.normal(size=(7,4))@((centroids-center)@basis)
        np.testing.assert_allclose(remove_span(x+shift,mean,q),remove_span(x,mean,q),atol=1e-12)


if __name__=='__main__':unittest.main()
