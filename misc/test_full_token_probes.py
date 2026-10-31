"""Small CPU checks for the full-token ridge solver and layer-by-layer inference."""
import unittest

import numpy as np
from sklearn.linear_model import Ridge
import torch

from misc.probe_full_tokens import advance_layer, fit_full_probe, select_layers, TASKS
from utils import metrics


class FullTokenProbesTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.rng = np.random.default_rng(7)
        self.x = (self.rng.normal(size=(30, 5, 8)) + 10).astype(np.float32)
        self.split = np.array(['A'] * 16 + ['B'] * 7 + ['D'] * 7)

    def test_matches_sklearn_including_intercept_and_selection(self):
        alphas = [.01, 1., 100.]
        for task in TASKS:
            if task == 'direction':
                angles = self.rng.uniform(-np.pi, np.pi, len(self.x))
                y = np.column_stack((np.sin(angles), np.cos(angles)))
            else:
                y = self.rng.normal(size=(len(self.x), 1)) + 3
            record, coef, intercept, predictions = fit_full_probe(
                self.x, y, self.split, task, alphas, 'cpu', chunk_size=13)
            flat = self.x.reshape(len(self.x), -1).astype(float)
            a, b = self.split == 'A', self.split == 'B'
            fits = [Ridge(alpha=alpha, solver='svd').fit(flat[a], y[a]) for alpha in alphas]
            scores = [metrics(y[b], fit.predict(flat[b]).reshape(y[b].shape), task) for fit in fits]
            best = int(np.argmin([score['mae'] for score in scores]))
            expected = fits[best].predict(flat).reshape(y.shape)
            self.assertEqual(record['alpha'], alphas[best])
            np.testing.assert_allclose(predictions, expected, atol=1e-9, rtol=1e-8)
            np.testing.assert_allclose(flat @ coef.T + intercept, expected, atol=1e-9, rtol=1e-8)
            # Changing test labels cannot affect the chosen alpha or fitted readout.
            y[self.split == 'D'] = np.nan
            other, coef2, intercept2, _ = fit_full_probe(
                self.x, y, self.split, task, alphas, 'cpu', chunk_size=9)
            self.assertEqual(record['alpha'], other['alpha'])
            np.testing.assert_allclose(coef, coef2, atol=1e-9)
            np.testing.assert_allclose(intercept, intercept2, atol=1e-9)

    def test_retains_information_that_pooling_discards(self):
        signal = self.rng.normal(size=30).astype(np.float32)
        x = np.stack([signal, -signal], axis=1)[..., None]
        y = signal.astype(float)[:, None]
        np.testing.assert_array_equal(x.mean(1), 0.)
        _, coef, _, predictions = fit_full_probe(
            x, y, self.split, 'speed', [.0001], 'cpu', chunk_size=1)
        self.assertGreater(coef[0, 0], 0)
        self.assertLess(coef[0, 1], 0)
        np.testing.assert_allclose(predictions, y, atol=1e-5)

    def test_layer_selection_uses_validation_only(self):
        results = {t: {'layers': [{'validation': {'r2': r, 'mae': 1 - r},
                                  'test': {'r2': -i}}
                                 for i, r in enumerate([.6, .9, .7])]} for t in TASKS}
        self.assertEqual(select_layers(results), {t: 2 for t in TASKS})
        results['speed']['layers'][1]['validation']['r2'] = .1
        results['speed']['layers'][1]['validation']['mae'] = 1.
        self.assertEqual(select_layers(results), {t: 3 for t in TASKS})

    def test_layerwise_execution_matches_normal_encoder(self):
        from transformers import VJEPA2Config, VJEPA2Model

        torch.manual_seed(7)
        config = VJEPA2Config(
            image_size=32, crop_size=32, patch_size=16, frames_per_clip=4,
            hidden_size=64, num_attention_heads=4, num_hidden_layers=3,
            pred_hidden_size=64, pred_num_attention_heads=4, pred_num_hidden_layers=1,
            mlp_ratio=2., pred_mlp_ratio=2., attn_implementation='sdpa')
        model = VJEPA2Model(config).eval().requires_grad_(False)
        pixels = torch.randn(3, 4, 3, 32, 32)
        expected = []
        handles = [block.register_forward_hook(
            lambda module, inputs, output: expected.append(output[0].detach().clone()))
            for block in model.encoder.layer]
        with torch.inference_mode():
            model.get_vision_features(pixels)
            states = model.encoder.embeddings(pixels).numpy().copy()
        for handle in handles:
            handle.remove()
        for block, reference in zip(model.encoder.layer, expected):
            advance_layer(block, states, 'cpu', batch_size=2)
            np.testing.assert_allclose(states, reference.numpy(), atol=2e-6, rtol=2e-6)


if __name__ == '__main__':
    unittest.main()
