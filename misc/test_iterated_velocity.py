import unittest

import numpy as np
import torch
from sklearn.linear_model import Ridge

from misc.iterated_velocity import constrain_inverse, fit_chunk, ridge_inverse
from misc.evaluate_iterated_velocity import erase, condition_list


class IteratedVelocityTests(unittest.TestCase):
    def test_inverse_rank_updates_equal_direct_constrained_ridge(self):
        rng = np.random.default_rng(21)
        xa = torch.tensor(rng.normal(size=(2, 30, 12)), dtype=torch.float64)
        xa -= xa.mean(1, keepdim=True)
        basis = torch.linalg.qr(torch.tensor(rng.normal(size=(2, 12, 8)), dtype=torch.float64)).Q
        empty = basis[:, :, :0]
        h = ridge_inverse(xa, empty, [.01, 1., 100.])
        for k in range(0, 8, 2):
            h = constrain_inverse(h, basis[:, :, k:k+2])
            direct = ridge_inverse(xa, basis[:, :, :k+2], [.01, 1., 100.])
            torch.testing.assert_close(h, direct, atol=1e-10, rtol=1e-8)
        # Padded zero constraint is harmless.
        same = constrain_inverse(h, torch.zeros_like(basis[:, :, :2]))
        torch.testing.assert_close(h, same, atol=0, rtol=0)

    def test_sequences_match_explicit_project_fit_and_ignore_cd(self):
        rng = np.random.default_rng(37)
        x = rng.normal(size=(72, 2, 16)) + 2
        y = x[:, 0, :2] + .2*rng.normal(size=(72, 2))
        split = np.array(['A']*36+['B']*20+['C']*8+['D']*8)
        pos = rng.normal(size=(2, 2, 16))
        alphas = [.001, .1, 10.]
        saved = fit_chunk(x, y, split, pos, max_rank=6, alphas=alphas, device='cpu')
        ya = y[split=='A']; scale = ya.std(0)
        for token in range(2):
            q = saved['position_basis'][token]
            centered = x[:, token]-x[split=='A', token].mean(0)
            for step in range(len(saved['active'])):
                if not saved['active'][step, token]:
                    continue
                projected = centered-(centered@q)@q.T
                candidates = [Ridge(alpha=a, solver='svd').fit(projected[split=='A'], ya/scale)
                              for a in alphas]
                errors = [np.abs(p.predict(projected[split=='B'])-y[split=='B']/scale).mean()
                          for p in candidates]
                best = int(np.argmin(errors))
                self.assertEqual(saved['alpha'][step, token], alphas[best])
                expected = candidates[best].coef_*scale[:, None]
                expected -= (expected@q)@q.T
                np.testing.assert_allclose(saved['coef'][step, token], expected, atol=1e-8)
                r = saved['rank_before'][step, token]+saved['removed_rank'][step, token]
                q = np.concatenate((saved['position_basis'][token], saved['velocity_basis'][token, :, :r]), axis=1)
        changed_x, changed_y = x.copy(), y.copy()
        changed_x[np.isin(split, ['C', 'D'])] = np.nan
        changed_y[np.isin(split, ['C', 'D'])] = np.nan
        again = fit_chunk(changed_x, changed_y, split, pos, max_rank=6, alphas=alphas, device='cpu')
        for name in ('coef', 'alpha', 'eligible', 'final_rank', 'velocity_basis'):
            np.testing.assert_allclose(saved[name], again[name], atol=0, rtol=0)
        self.assertLess(saved['orthogonality_max_error'], 1e-10)
        self.assertTrue(np.all(saved['final_rank'] <= 6))

    def test_erasure_preserves_position_and_unselected_coordinates(self):
        rng = np.random.default_rng(71)
        q = np.linalg.qr(rng.normal(size=(3,12,6)))[0]
        position = torch.tensor(q[:,:,:2],dtype=torch.float64)
        velocity = torch.tensor(q[:,:,2:],dtype=torch.float64)
        h = torch.tensor(rng.normal(size=(5,3,12)),dtype=torch.float64)
        mean = torch.tensor(rng.normal(size=(3,12)),dtype=torch.float64)
        edited = erase(h,mean,velocity)
        torch.testing.assert_close(torch.einsum('btc,tcr->btr',edited-h,position),
                                   torch.zeros((5,3,2),dtype=torch.float64),atol=1e-12,rtol=0)
        torch.testing.assert_close(torch.einsum('btc,tcr->btr',edited-mean,velocity),
                                   torch.zeros((5,3,4),dtype=torch.float64),atol=1e-12,rtol=0)
        torch.testing.assert_close(erase(edited,mean,velocity),edited,atol=1e-12,rtol=0)

    def test_sum_of_orthogonal_probe_edits_satisfies_every_probe(self):
        rng = np.random.default_rng(81)
        q = np.linalg.qr(rng.normal(size=(20,8)))[0]
        weights = [np.array([[1.,.2],[.3,2.]])@q[:,2*j:2*j+2].T for j in range(1,4)]
        operator = sum(np.linalg.pinv(w) for w in weights)
        for w in weights:
            np.testing.assert_allclose(w@operator,np.eye(2),atol=1e-12)
        np.testing.assert_allclose(q[:,:2].T@operator,0,atol=1e-12)
        stacked = np.concatenate(weights)
        direct = np.linalg.pinv(stacked)@np.tile(np.eye(2),(3,1))
        np.testing.assert_allclose(operator,direct,atol=1e-12)


if __name__ == '__main__':
    unittest.main()
