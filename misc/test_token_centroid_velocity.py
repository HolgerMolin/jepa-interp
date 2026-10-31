import unittest

import numpy as np
from sklearn.decomposition import PCA
import torch

from misc.token_centroid_velocity import (fit_token, curve_value, curve_difference,
                                     TokenCurves, random_edit)
from utils import fit_centroid_spline


class TokenCentroidTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        rng = np.random.default_rng(301)
        cls.sa = np.linspace(.25, 4., 80)
        cls.sb = np.linspace(.26, 3.99, 32)
        cls.xa, cls.xb, cls.models, cls.selections, cls.q = [], [], [], [], []
        for token in range(3):
            w = rng.normal(size=(3, 14))
            def features(speed):
                return np.column_stack([speed, np.sin(speed*(token+1)), speed**2])@w+.3*rng.normal(size=(len(speed),14))
            a, b = features(cls.sa), features(cls.sb)
            model, selection = fit_token(a, b, cls.sa, cls.sb, dimensions=6, knot_options=[4,6,8])
            cls.xa.append(a); cls.xb.append(b); cls.models.append(model); cls.selections.append(selection)
            cls.q.append(np.linalg.qr(rng.normal(size=(14,2)))[0])
        cls.q = np.stack(cls.q)

    def test_against_independent_sklearn_pca_and_original_helper(self):
        for a, b, model, selection in zip(self.xa, self.xb, self.models, self.selections):
            pca = PCA(n_components=6, svd_solver='full').fit(a)
            np.testing.assert_allclose(model['components'].T@model['components'], pca.components_.T@pca.components_, atol=1e-10)
            candidates = []
            for count in [4,6,8]:
                curve, _, _, _ = fit_centroid_spline(pca.transform(a), self.sa, 'speed', count, [.25,4.])
                candidates.append(np.square(pca.transform(b)-curve(self.sb)).mean())
            self.assertEqual(selection['selected']['knots'], [4,6,8][np.argmin(candidates)])
            np.testing.assert_allclose([row['B_PCA_reconstruction_mse'] for row in selection['candidates']], candidates, atol=1e-10)
            ref, _, _, _ = fit_centroid_spline(pca.transform(a), self.sa, 'speed', selection['selected']['knots'], [.25,4.])
            s = np.linspace(.75,3.5,17)
            np.testing.assert_allclose(curve_difference(model,s,.25), (ref(s+.25)-ref(s))@pca.components_, atol=1e-10)

    def test_fits_use_only_training_data_and_do_not_mutate_inputs(self):
        a, b = self.xa[0].copy(), self.xb[0].copy()
        changed, _ = fit_token(a, b+5., self.sa, self.sb, dimensions=6, knot_options=[4])
        original, _ = fit_token(a, b, self.sa, self.sb, dimensions=6, knot_options=[4])
        for key in ['components','mean','coefficients','centroids']:
            np.testing.assert_array_equal(changed[key], original[key])
        np.testing.assert_array_equal(a, self.xa[0]); np.testing.assert_array_equal(b, self.xb[0])

    def test_exact_residual_identity_reverse_and_position_protection(self):
        s = np.linspace(.75,3.5,19)
        h = self.xa[0][:len(s)]
        model = self.models[0]
        for q in [None, self.q[0]]:
            delta = curve_difference(model,s,.5,q)
            np.testing.assert_allclose((h+delta)-curve_value(model,s+.5,q), h-curve_value(model,s,q), atol=1e-12)
            np.testing.assert_array_equal(curve_difference(model,s,0.,q), np.zeros_like(h))
            np.testing.assert_allclose(delta, -curve_difference(model,s+.5,-.5,q), atol=1e-12)
            if q is not None:
                np.testing.assert_allclose(delta@q, 0., atol=1e-12)
        raw = curve_difference(model,s,.5)
        components = model['components']
        np.testing.assert_allclose(raw-(raw@components.T)@components, 0., atol=1e-12)

    def test_batched_token_curves_match_independent_scipy(self):
        # Force different knot counts to exercise all grouping and boundary cases.
        models = [fit_token(a,b,self.sa,self.sb,dimensions=6,knot_options=[k])[0]
                  for a,b,k in zip(self.xa,self.xb,[4,6,8])]
        curves = TokenCurves(models,self.q,device='cpu')
        s = np.array([.25,.5,.75,1.5,2.5,3.5,4.])
        expected = np.stack([curve_value(m,s) for m in models],axis=1)
        np.testing.assert_allclose(curves.values(s).numpy(),expected,atol=1e-11)
        source = np.array([.75,1.,2.,3.5])
        for size in [-.5,-.25,0.,.25,.5]:
            for protect in [False,True]:
                reference = np.stack([curve_difference(m,source,size,self.q[t] if protect else None) for t,m in enumerate(models)],axis=1)
                actual = curves.difference(source,size,protect).numpy()
                np.testing.assert_allclose(actual,reference,atol=1e-11)
                if size==0.: np.testing.assert_array_equal(actual,np.zeros_like(actual))
        self.assertGreater(np.linalg.norm(expected[:,0]-expected[:,1]),1.)

    def test_random_controls_match_each_token_norm_and_protection(self):
        rng = np.random.default_rng(302)
        curves = TokenCurves(self.models,self.q,device='cpu')
        for protect in [False,True]:
            direction = rng.normal(size=(3,14))
            if protect:
                direction -= np.einsum('tr,tcr->tc',np.einsum('tc,tcr->tr',direction,self.q),self.q)
            direction /= np.linalg.norm(direction,axis=-1,keepdims=True)
            delta = curves.difference(np.array([1.,2.,3.]),.5,protect)
            control = random_edit(delta,torch.tensor(direction),.5).numpy()
            np.testing.assert_allclose(np.linalg.norm(control,axis=-1),np.linalg.norm(delta.numpy(),axis=-1),atol=1e-12)
            if protect:
                np.testing.assert_allclose(np.einsum('btc,tcr->btr',control,self.q),0.,atol=1e-12)

    def test_extrapolation_and_nonfinite_source_are_rejected(self):
        curves = TokenCurves(self.models,self.q,device='cpu')
        for bad in [[.2],[4.1],[np.nan],[np.inf]]:
            with self.assertRaises(ValueError): curve_value(self.models[0],bad)
            with self.assertRaises(ValueError): curves.values(bad)
        with self.assertRaises(ValueError): curves.difference([3.75],.5)


if __name__ == '__main__':
    unittest.main()
