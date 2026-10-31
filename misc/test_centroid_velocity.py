import unittest
import numpy as np
from sklearn.decomposition import PCA
from utils import fit_centroid_spline,replace_pca_coordinates
from misc.centroid_velocity import lift,pooled_edit


class CentroidTests(unittest.TestCase):
    def test_exact_original_recipe_and_lift(self):
        rng=np.random.default_rng(17)
        h=rng.normal(size=(40,7,12)); x=h.mean(1)
        pca=PCA(n_components=5,svd_solver='full').fit(x[:25])
        target=rng.normal(size=(40,5))
        delta=pooled_edit(x,target,pca.mean_,pca.components_)
        edited=lift(h,delta); expected=replace_pca_coordinates(x,pca,target)
        np.testing.assert_allclose(edited.mean(1),expected,atol=1e-12)
        np.testing.assert_allclose(pca.transform(edited.mean(1)),target,atol=1e-12)
        np.testing.assert_allclose(edited-edited.mean(1,keepdims=True),h-x[:,None],atol=1e-12)
        extra=rng.normal(size=h.shape);extra-=extra.mean(1,keepdims=True)
        uniform=np.broadcast_to(delta[:,None],h.shape)
        np.testing.assert_allclose(np.square(uniform+extra).sum(),np.square(uniform).sum()+np.square(extra).sum(),atol=1e-10)

    def test_centroid_interpolation_and_source_control(self):
        rng=np.random.default_rng(18)
        labels=np.linspace(.25,4.,80)
        z=rng.normal(size=(80,6))
        curve,anchors,centroids,counts=fit_centroid_spline(z,labels,'speed',8,[.25,4.])
        np.testing.assert_allclose(curve(anchors),centroids,atol=1e-12)
        assignment=np.abs(labels[:,None]-anchors).argmin(1)
        for j in range(8): np.testing.assert_allclose(centroids[j],z[assignment==j].mean(0))
        self.assertEqual(counts.sum(),80)
        self.assertTrue(np.isnan(curve(5.)).all())
        source=curve(labels)-z
        self.assertGreater(np.linalg.norm(source),0.)
        target=curve(labels[12:60]+.25)-z[12:60]
        np.testing.assert_allclose(target-source[12:60],curve(labels[12:60]+.25)-curve(labels[12:60]),atol=1e-12)
        rotation=np.linalg.qr(rng.normal(size=(6,6)))[0]
        np.testing.assert_allclose(np.linalg.norm(target@rotation,axis=1),np.linalg.norm(target,axis=1),atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm((target-source[12:60])@rotation,axis=1),np.linalg.norm(target-source[12:60],axis=1),atol=1e-12)


if __name__=='__main__':unittest.main()
