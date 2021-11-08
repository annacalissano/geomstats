r"""Unit tests for the space of PSD matrices of rank k."""

import warnings

import tests.helper as helper

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.rank_k_psd_matrices import PSDMatrices


class TestPSDMatricesRankK(geomstats.tests.TestCase):
    r"""Test of PSD Matrices Rank k methods."""

    def setUp(self):
        r"""Set up the test."""
        warnings.simplefilter("ignore", category=ImportWarning)

        gs.random.seed(1234)

        self.n = 3
        self.k = 2
        self.space = PSDMatrices(self.n, self.k)

    def test_belongs(self):
        r"""Test of belongs method."""
        psd_n_k = self.space
        mat_not_psd_n_k = gs.array(
            [
                [0.27053942, -0.34773248, 0.2672531],
                [-0.34773248, 0.77543347, 0.09687998],
                [0.2672531, 0.09687998, 0.85442487],
            ]
        )
        mat_psd_n_k = gs.array([[1.0, 0.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 12.0]])
        result = psd_n_k.belongs(mat_not_psd_n_k)
        expected = False
        self.assertFalse(result, expected)

        result = sym_n.belongs(mat_psd_n_k)
        expected = True
        self.assertTrue(result, expected)

    def test_projection_and_belongs(self):
        shape = (2, self.n, self.n)
        result = helper.test_projection_and_belongs(self.space, shape)
        for res in result:
            self.assertTrue(res)

    def test_random_and_belongs(self):
        mat = self.space.random_point(4)
        result = self.space.belongs(mat)
        self.assertTrue(gs.all(result))
