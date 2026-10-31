import unittest

import numpy as np
import torch
from sklearn.linear_model import Ridge

from misc.velocity_causal import disk_positions, ridge_path, spatial_features
from misc.velocity_intervention import constrained_operator, effect_metrics


class VelocityCausalTests(unittest.TestCase):
    def test_centroids_follow_image_coordinates(self):
        frames = np.full((2, 20, 24, 3), 30, dtype=np.uint8)
        frames[0, 4:10, 10:16] = [20, 60, 220]
        frames[1, 7:13, 13:19] = [220, 100, 40]
        np.testing.assert_allclose(disk_positions(frames), [[12.5, 6.5], [15.5, 9.5]])
        with self.assertRaises(ValueError):
            disk_positions(np.zeros_like(frames))

    def test_spatial_pooling_keeps_time_height_width_order(self):
        grid = torch.arange(4*16*16, dtype=torch.float32).reshape(1, 4, 16, 16, 1)
        tokens = grid.expand(-1, -1, -1, -1, 1024).reshape(1, 1024, 1024)
        result = spatial_features(tokens).reshape(1, 4, 4, 4, 1024)
        for t in range(4):
            for y in range(4):
                for x in range(4):
                    expected = grid[0, t, y*4:y*4+4, x*4:x*4+4, 0].mean()
                    torch.testing.assert_close(result[0, t, y, x], expected.expand(1024))

    def test_dual_ridge_matches_sklearn_and_ignores_test_labels(self):
        rng = np.random.default_rng(11)
        x = rng.normal(size=(25, 40)) + 3
        y = rng.normal(size=(25, 2))
        a, b = np.arange(18), np.arange(18, 25)
        alphas = [.01, 1, 100]
        result = ridge_path(x, y, a, b, alphas)
        for i, alpha in enumerate(alphas):
            expected = Ridge(alpha=alpha, solver='cholesky').fit(x[a], y[a]).predict(x[b])
            np.testing.assert_allclose(result[i], expected, atol=1e-9)
        y[b] = np.nan
        np.testing.assert_allclose(ridge_path(x, y, a, b, alphas), result, atol=0, rtol=0)

    def test_velocity_edit_preserves_position_and_is_minimum_norm(self):
        rng = np.random.default_rng(16)
        scale = np.array([2., 3., 50., 60.])
        w = rng.normal(size=(3, 4, 20))*scale[None, :, None]
        op = constrained_operator(w, scale)
        delta_v = np.array([.25, -.5])
        delta_h = op @ delta_v
        achieved = np.einsum('toc,tc->to', w, delta_h)
        expected = np.broadcast_to(np.r_[delta_v, 0, 0], achieved.shape)
        np.testing.assert_allclose(achieved, expected, atol=1e-12)
        for t in range(3):
            reference = np.linalg.pinv(w[t]) @ expected[t]
            np.testing.assert_allclose(delta_h[t], reference, atol=1e-12)

    def test_time_slope_separates_velocity_from_position_shift(self):
        times = np.array([1.5, 3.5, 5.5, 7.5])/24
        v = np.array([[8., 0.], [0., -10.], [6., 8.]])
        shift = times[None, :, None]*v[:, None]
        measured = effect_metrics(shift, v, times)
        self.assertAlmostEqual(measured['velocity_slope_gain']['mean'], 1.)
        self.assertAlmostEqual(measured['displacement_gain']['mean'], 1.)
        self.assertLess(measured['position_intercept_pixels']['mean'], 1e-12)
        constant = np.repeat(np.array([[2., 3.], [-1., 4.], [3., 2.]])[:, None], 4, axis=1)
        confound = effect_metrics(constant, v, times)
        self.assertAlmostEqual(confound['velocity_slope_gain']['mean'], 0.)
        self.assertGreater(confound['position_intercept_pixels']['mean'], 3.)


if __name__ == '__main__':
    unittest.main()
