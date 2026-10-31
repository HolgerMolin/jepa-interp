"""Verify independent token fits against the existing scikit-learn methodology."""
import unittest

import numpy as np
from sklearn.linear_model import Ridge
import torch

from misc.probe_each_token import ALPHAS, evaluate_tokens, fit_token_batch, scores
from utils import bootstrap_interval, metrics


class EachTokenTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.rng = np.random.default_rng(100)
        self.split = np.array(['A'] * 20 + ['B'] * 8 + ['C'] * 3 + ['D'] * 9)

    def test_independent_fits_match_existing_methodology(self):
        for channels in (12, 40):
            x = self.rng.normal(size=(40, 4, channels)).astype(np.float32)
            x += np.arange(4, dtype=np.float32)[None, :, None] * 10
            for task in ('direction', 'speed', 'acceleration'):
                if task == 'direction':
                    theta = self.rng.uniform(-np.pi, np.pi, len(x))
                    y = np.column_stack([np.sin(theta), np.cos(theta)])
                else:
                    y = self.rng.normal(size=(len(x), 1)) + 3
                result = fit_token_batch(x, y, self.split, task, device='cpu')
                a, b, d = (self.split == s for s in 'ABD')
                for token in range(4):
                    flat = x[:, token].astype(float)
                    candidates = [Ridge(alpha=alpha, solver='cholesky').fit(flat[a], y[a])
                                  for alpha in ALPHAS]
                    val = [metrics(y[b], p.predict(flat[b]), task) for p in candidates]
                    best = min(range(len(ALPHAS)), key=lambda i: val[i]['mae'])
                    self.assertEqual(result['alpha'][token], ALPHAS[best])
                    np.testing.assert_allclose(result['validation_grid'][token],
                                               [[v['r2'], v['mae']] for v in val], atol=1e-6)
                    expected = candidates[best].predict(flat[d]).reshape(y[d].shape)
                    np.testing.assert_allclose(result['test_prediction'][token], expected, atol=1e-7)
                    direct = flat[d] @ result['coef'][token].T + result['intercept'][token]
                    np.testing.assert_allclose(direct, expected, atol=1e-7)
                    # Running this token alone must give exactly the same fitted model.
                    single = fit_token_batch(x[:, token:token + 1], y, self.split, task, device='cpu')
                    np.testing.assert_allclose(single['coef'][0], result['coef'][token], atol=1e-8)
                # Reserved/test labels and reserved features cannot affect a fitted probe.
                corrupted_y = y.copy()
                corrupted_y[(self.split == 'C') | d] = np.nan
                changed = x.copy()
                changed[self.split == 'C'] = 1e6
                again = fit_token_batch(changed, corrupted_y, self.split, task, device='cpu')
                np.testing.assert_array_equal(result['alpha'], again['alpha'])
                np.testing.assert_allclose(result['coef'], again['coef'], atol=1e-9)

    def test_bootstrap_matches_existing_clip_bootstrap(self):
        for task in ('direction', 'speed', 'acceleration'):
            if task == 'direction':
                angle = self.rng.uniform(-np.pi, np.pi, 25)
                y = np.column_stack((np.sin(angle), np.cos(angle)))
            else:
                y = self.rng.normal(size=(25, 1))
            prediction = y[None] + self.rng.normal(scale=.3, size=(3, *y.shape))
            result = evaluate_tokens(y, prediction, task, device='cpu')
            for token in range(3):
                expected = metrics(y, prediction[token], task)
                ci = bootstrap_interval(y, prediction[token], task)
                for metric in ('r2', 'mae'):
                    np.testing.assert_allclose(result[f'test_{metric}'][token], expected[metric], atol=1e-10)
                    np.testing.assert_allclose(result[f'test_{metric}_ci95'][token], ci[metric], atol=1e-10)

    def test_direction_wrap_and_zero_vector(self):
        angles = np.deg2rad([179., -179., 0., 90.])
        y = np.column_stack((np.sin(angles), np.cos(angles)))
        prediction = y.copy()
        prediction[:2] = prediction[[1, 0]]
        prediction[2:] = 0.
        result = scores(torch.tensor(y), torch.tensor(prediction)[None], 'direction')[0].numpy()
        expected = metrics(y, prediction, 'direction')
        np.testing.assert_allclose(result, [expected['r2'], expected['mae']], atol=1e-10)


if __name__ == '__main__':
    unittest.main()
