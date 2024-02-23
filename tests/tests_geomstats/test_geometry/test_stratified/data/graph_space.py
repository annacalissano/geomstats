from geomstats.test.data import TestData
from geomstats.test.test_case import np_backend

from ...data.matrices import MatricesTestData
from .quotient import AlignerAlgorithmCmpTestData, QuotientMetricWithArrayTestData

IS_NOT_NP = not np_backend()


class GraphAlignerCmpTestData(AlignerAlgorithmCmpTestData):
    N_RANDOM_POINTS = [1, 2]
    trials = 5


class PointToGeodesicAlignerTestData(TestData):
    skip_all = IS_NOT_NP
    fail_for_not_implemented_errors = False

    tolerances = {"dist_along_geodesic_is_zero": {"atol": 1e-2}}

    def align_vec_test_data(self):
        return self.generate_vec_data()

    def dist_vec_test_data(self):
        return self.generate_vec_data()

    def dist_along_geodesic_is_zero_test_data(self):
        return self.generate_random_data()


class GraphSpaceTestData(MatricesTestData):
    skip_all = IS_NOT_NP


class GraphSpaceQuotientMetricTestData(QuotientMetricWithArrayTestData):
    skip_all = IS_NOT_NP
