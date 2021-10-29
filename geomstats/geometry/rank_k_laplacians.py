"""The manifold of Positive Semi Definite matrices of rank k."""

import math

import geomstats.backend as gs
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.spd_matrices import (
    SPDMatrices,
    SPDMetricBuresWasserstein,
    SPDMetricAffine,
    SPDMetricEuclidean,
    SPDMetricLogEuclidean,
)
from geomstats.geometry.symmetric_matrices import SymmetricMatrices
import numpy as np
import copy

# ANNA add the function fill_diagonal to the gs


class RankKLaplacians(Manifold):
    """Class for the manifold of discrete laplacian matrices of a given rank k.

    Parameters
    ----------
    n : int
        Integer representing the shape of the matrices: n x n.
    k: int
        Integer representing the rank of the matrix (k<n).

    References
    ----------
    .. [GS2017] Ginestet, C. E., Li, J., Balachandran, P., Rosenberg, S., & Kolaczyk, E. D. (2017).
     Hypothesis testing for network data in functional neuroimaging.
     The Annals of Applied Statistics, 725-750.
    """

    def __init__(
        self,
        n,
        k,
        metric=None,
        default_point_type="matrix",
        default_coords_type="intrinsic",
        **kwargs
    ):
        super(Manifold, self).__init__(**kwargs)
        self.n = n
        self.dim = int(k * n - k * (k + 1) / 2)
        self.default_point_type = default_point_type
        self.default_coords_type = default_coords_type
        self.metric = metric
        self.rank = k
        self.sym = SymmetricMatrices(self.n)

    # ANNA - how do we create copies? on graphspace i was using copy.deepcopy()

    def belongs(self, point, atol=gs.atol):
        """Check if a matrix is a laplacian matrix

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Matrix to be checked.
        atol : float
            Tolerance.
            Optional, default: backend atol.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean denoting if mat is a Laplacian matrix of rank k.
        """
        mat=copy.deepcopy(point)
        is_symmetric = self.sym.belongs(mat, atol)
        eigval, eigvec = gs.linalg.eig(mat)
        is_semipositive = gs.all(eigval > -atol, axis=-1)
        is_rankk = gs.linalg.matrix_rank(mat) == self.rank

        is_sumtozero = mat.sum(axis=0).sum() == 0
        np.fill_diagonal(mat, 0)
        is_neg_out_diag = gs.sum(gs.array(mat) <= 0) == self.n * self.n
        belongs = gs.logical_and(
            gs.logical_and(
                gs.logical_and(gs.logical_and(is_symmetric, is_semipositive), is_rankk),
                is_sumtozero,
            ),
            is_neg_out_diag,
        )

        return belongs[0, 0]

    # ANNA qui rivedi projection per psd per vettori
    def projection(self, point):
        """Project a matrix to the space of L(n,k) matrices

        The input matrix is turned into symmetric,
        projected onto the positive orthant to ensure positive entries
        transformed into an adjecency matrix posing as null the diagonal values
        and finally transformed into a Laplacian

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Matrix to project.

        Returns
        -------
        projected: array-like, shape=[..., n, n]
            Laplacian matrix of rank k.
        """

        to_sym = SymmetricMatrices(self.n).projection(point)
        to_lap = [gs.eye(self.n) * i.sum(axis=0) - i - gs.eye(self.n) * i.diagonal() for i in to_sym]
        to_lap[0 : (self.n - self.rank)] = [0] * (self.n - self.rank)
        eigvals, eigvecs = gs.linalg.eigh(np.array(to_lap))
        regularized = gs.where(eigvals < 0, 0, eigvals)
        regularized[:, 0: (5 - 3)] = 0
        reconstruction = gs.einsum("...ij,...j->...ij", eigvecs, regularized)
        return Matrices.mul(reconstruction, Matrices.transpose(eigvecs))

    def random_point(self, n_samples=1, bound=1.0):
        """Sample in L(n,k) by sampling symmetric matrix
        - with a uniform distribution in a box -
        and transform it into a laplacian

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Bound of the interval in which to sample in the tangent space.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., n, n]
            Points sampled in L(n,k).
        """
        size = self.n
        if n_samples != 1:
            size = (n_samples,) + (self.n, self.n)
        sample = gs.random.rand(*size)

        if n_samples > 1:
            lap_mat = [self.projection(i) for i in sample]
        else:
            lap_mat = [self.projection(sample)]
        return lap_mat

    # ANNA add the correct citation of Yann's work

    def is_tangent(self, vector, base_point):
        """Check if the vector belongs to the tangent space at the input point.

        Parameters
        ----------
        vector : array-like, shape=[..., n, n]
            Matrix to check if it belongs to the tangent space.
        base_point : array-like, shape=[..., n, n]
            Base point of the tangent space.
            Optional, default: None.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean denoting if vector belongs to tangent space at base_point.
        References
        ----------
        .. [TP2019] Thanwerdas, Pennec. "Is affine-invariance well defined on
          SPD matrices? A principled continuum of metrics" Proc. of GSI, 2019.
          https://arxiv.org/abs/1906.01349
        """

        vector_sym = [
            vector if self.sym.belongs(vector) else self.sym.projection(vector)
        ][0]
        r, delta, rt = gs.linalg.svd(base_point)
        rort = r[:, self.n - self.rank : self.n]
        rort_t = rt[self.n - self.rank : self.n, :]
        check = gs.matmul(
            gs.matmul(gs.matmul(rort, rort_t), vector_sym), gs.matmul(rort, rort_t)
        )
        if (
            gs.logical_and(
                gs.less_equal(check, -gs.atol), gs.greater(check, gs.atol)
            ).sum()
            == 0
        ):
            return True
        else:
            return False

    def to_tangent(self, vector, base_point):
        """Project the input vector to the tangent space of PSD(n,k) at base_point.

        Parameters
        ----------
        vector : array-like, shape=[..., n, n]
            Matrix to check if it belongs to the tangent space.
        base_point : array-like, shape=[..., n, n]
            Base point of the tangent space.
            Optional, default: None.

        Returns
        -------
        tangent : array-like, shape=[...,n,n]
            Projection of the tangent vector at base_point.
        """
        if self.is_tangent(vector, base_point):
            return vector
        else:
            vector_sym = [
                vector if self.sym.belongs(vector) else self.sym.projection(vector)
            ][0]
            r, delta, rt = gs.linalg.svd(base_point)
            rort = r[:, self.n - self.rank : self.n]
            rort_t = rt[self.n - self.rank : self.n, :]
            return (
                gs.matmul(
                    gs.matmul(gs.matmul(rort, rort_t), vector_sym),
                    gs.matmul(rort, rort_t),
                )
                + vector_sym
            )


PSDMetricBuresWasserstein = SPDMetricBuresWasserstein

PSDMetricEuclidean = SPDMetricEuclidean

PSDMetricLogEuclidean = SPDMetricLogEuclidean

PSDMetricAffine = SPDMetricAffine


class Laplacians(RankKLaplacians):
    r"""Class for the psd matrices. The class is recirecting to the correct embedding manifold.
    The stratum PSD rank k if the matrix is not full rank
    The top stratum SPD if the matrix is full rank
    The whole stratified space of PSD if no rank is specified

    Parameters
    ----------
    n : int
        Integer representing the shapes of the matrices : n x n.
    k : int
        Integer representing the shapes of the matrices : n x n.
    """

    def __new__(
        cls,
        n,
        k=None,
        metric=None,
        default_point_type="matrix",
        default_coords_type="intrinsic",
    ):
        if k == n:
            raise NotImplementedError("Carefull! It is not a laplacian matrix!")
        elif n > k:
            return RankKLaplacians(n, k)
