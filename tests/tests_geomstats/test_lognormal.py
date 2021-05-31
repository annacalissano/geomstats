"""Unit tests for the LogNormal Sampler."""

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.geometry.euclidean import  Euclidean
from geomstats.sampling.lognormal import LogNormal
from geomstats.geometry.hypersphere import Hypersphere


class TestLogNormal(geomstats.tests.TestCase):
    """Class defining the LogNormal tests."""

    def setUp(self):
        """Set up  the test"""
        self.n = 3
        self.spd_cov_n = (self.n * (self.n + 1)) // 2
        self.samples = 5
        self.SPDManifold = SPDMatrices(self.n)
        self.Euclidean = Euclidean(self.n)

    def test_euclidean_belongs(self):
        """Test if the samples belong to Euclidean Space"""
        mean = gs.zeros(self.n)
        cov = gs.eye(self.n)
        LogNormalSampler = LogNormal(self.Euclidean, mean, cov)
        data = LogNormalSampler.sample(self.samples)
        self.Eucludean.belongs(data)

    def test_spd_belongs(self):
        """Test if the samples to SPD Manifold"""
        mean = 2 * gs.eye(self.n)
        cov = gs.eye(self.spd_cov_n)
        LogNormalSampler = LogNormal(self.SPDManifold, mean, cov)
        data = LogNormalSampler.sample(self.samples)
        self.SPDManifold.belongs(data)

    def test_euclidean_frechet_mean(self):
        """Test if the frechet mean of the samples is close to mean"""
        mean = gs.zeros(self.n)
        cov = gs.eye(self.n) / self.n
        data = LogNormal(self.Euclidean, mean, cov).sample(100)
        fm = data.mean(axis=0)

        expected = mean
        result = fm
        self.assertAllClose(result, expected)

    def test_spd_frechet_mean(self):
        """Test if the frechet mean of the samples is close to mean"""
        mean = gs.eye(self.n)
        cov = gs.eye(self.spd_cov_n) / self.spd_cov_n
        data = LogNormal(self.SPDManifold, mean, cov).sample(100)
        _fm = self.SPDManifold.logm(data).mean(axis=0)
        fm = self.SPDManifold.expm(_fm)

        expected = mean
        result = fm
        self.assertAllClose(result, expected)

    def test_error_handling(self):
        """Test if the erros are raised for invalid parameters"""

        mean = gs.eye(self.n)
        invalid_eu_mean = gs.zeros(self.n + 1)
        invalid_spd_mean = gs.zeros((self.n, self.n))
        invalid_cov = gs.eye((self.n, self.n))
        invalid_manifold = Hypersphere(dim=2)

        with self.assertRaises(ValueError):
            LogNormal(invalid_manifold, mean)
        with self.assertRaises(ValueError):
            LogNormal(self.Euclidean, invalid_eu_mean)
        with self.assertRaises(ValueError):
            LogNormal(self.SPDManifold, invalid_spd_mean)
        with self.assertRaises(ValueError):
            LogNormal(self.SPDManifold, mean, invalid_cov)
