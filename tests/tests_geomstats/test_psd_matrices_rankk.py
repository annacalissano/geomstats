"""Unit tests for the space of PSD matrices of rank k."""

import warnings

import tests.helper as helper

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.psd_matrices_rankk import PSDMatricesRankK


class TestPSDMatricesRankK(geomstats.tests.TestCase):
    """Test of PSD Matrices Rank k methods."""

    def setUp(self):
        """Set up the test."""
        warnings.simplefilter("ignore", category=ImportWarning)

        gs.random.seed(1234)

        self.n = 5
        self.k = 3
        self.space = PSDMatricesRankK(self.n, self.k)

    def test_belongs(self):
        """Test of belongs method."""
        psd_n_k = self.space
        mat_psd_n_k = gs.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]])
        mat_not_psd_n_k = gs.array([[1.0, 0.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]])
        result = psd_n_k.belongs(mat_psd_n_k)
        expected = True
        self.assertAllClose(result, expected)

        result = sym_n.belongs(mat_not_psd_n_k)
        expected = False
        self.assertAllClose(result, expected)

    def test_projection_and_belongs(self):
        shape = (2, self.n, self.n)
        result = helper.test_projection_and_belongs(self.space, shape)
        for res in result:
            self.assertTrue(res)

    def test_random_and_belongs(self):
        mat = self.space.random_point()
        result = self.space.belongs(mat)
        self.assertTrue(result)

    def test_dim(self):
        result = self.space.dim
        n = self.space.n
        expected = int(n * (n + 1) / 2)
        self.assertAllClose(result, expected)
