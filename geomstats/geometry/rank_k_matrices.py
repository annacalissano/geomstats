"""Module exposing the rank k euclidean matrices"""

import geomstats.backend as gs
from geomstats.geometry.base import Manifold
from geomstats.geometry.matrices import Matrices


class RankKMatrices(Manifold):
    """Class for the matrices with euclidean entries and rank k

    Parameters
    ----------
    n : int
        Integer representing the shape of the matrices: m x n
    m : int
        Integer representing the shape of the matrices: m x n
    k : int
        Integer representing the rank of the matrices

    """

    def __init__(
        self,
        m,
        n,
        k,
        metric=None,
        default_point_type="matrix",
        default_coords_type="intrinsic",
        **kwargs
    ):
        super(Manifold, self).__init__(**kwargs)
        self.dim = m * n
        self.shape = [m, n]
        self.default_point_type = default_point_type
        self.default_coords_type = default_coords_type
        self.metric = metric
        self.rank = k
        self.mat = Matrices(self.shape[0], self.shape[1])

    def belongs(self, point, atol=gs.atol):
        """Check if the matrix belongs to R_k^m*n

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Matrix to be checked.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean denoting if point is in R_*^m*n
        """
        has_right_size = self.mat.belongs(point)
        if gs.all(has_right_size):
            rank = gs.linalg.matrix_rank(point)
            if (rank == self.rank).all():
                return True
        return False

    def projection(self, point):
        r"""Project a matrix to the set of rank k matrices

        This is not a projection per se, but a regularization if the matrix input X
        is not already rank k, the eigenvalues and eigenvector decomposition
        is performed, then the : `math:`X + \epsilon [I_rank, 0]` is returned
        where :math:`\epsilon=gs.atol`

        Parameters
        ----------
        point : array-like, shape=[..., dim_embedding]
            Point in embedding manifold.

        Returns
        -------
        projected : array-like, shape=[..., dim_embedding]
            Projected point.
        """
        belongs = self.belongs(point)
        if not belongs:
            u, s, vh = gs.linalg.svd(point, full_matrices=False)
            s[self.rank : min(self.shape)] = 0
            smat = s * gs.eye(min(self.shape))
            return gs.dot(u, gs.dot(smat, vh))
        else:
            return point

    def random_point(self, n_samples=1):
        """Sample in R_k^m*n from the uniform distribution

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., n, n]
            Point sampled on R_k^m*n
        """
        raw_samples = gs.random.normal(size=(n_samples, self.shape[0], self.shape[1]))
        sample = [self.projection(i) for i in raw_samples]
        if n_samples == 1:
            return sample[0]
        else:
            return sample
    # ANNA this function is missing - check next week
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
        .. [VB2019] Vandereycken, Bart. "Low-rank matrix completion by Riemannian optimization."
        SIAM Journal on Optimization 23.2 (2013): 1214-1236.
        """

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

        -------
        References
        ----------
        .. [VB2019] Vandereycken, Bart. "Low-rank matrix completion by Riemannian optimization."
        SIAM Journal on Optimization 23.2 (2013): 1214-1236.
        """

        if self.is_tangent(vector, base_point):
            return vector
        else:
            u_full, s, v_full = gs.linalg.svd(base_point)
            u = u_full[:, 0:self.rank]
            v = v_full[0:self.rank, :]
            pu=gs.matmul(u,u.transpose())
            pv=gs.matmul(v.transpose(), v)
            return gs.matmul(gs.matmul(pu,vector),pv)+gs.matmul(gs.matmul((gs.eye(pu.shape)-pu), vector),pv)+ gs.matmul(pu,gs.matmul(vector,gs.eye(pu.shape)-pv))
