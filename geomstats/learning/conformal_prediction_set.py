"""Conformal Prediction Sets.

Lead author: Anna Calissano.
"""
import geomstats.backend as gs


class conformal_prediction_set(object):
    """Class for the Computing Conformal Prediction Sets on Metric Spaces.

    Implementing conformal prediction sets using conformal prediction with amplitude
    modulation described in [Calissano2022].

    Parameters
    ----------
    space : GraphSpace
        Metric Space where your data live.
    metric : GraphSpaceMetric
        Metric between elements of space.
    predrule : Prediction Method
        Method used to perform prediction on space.
    alpha : float
        Coverage of the prediction interval
    calibration_size : float
        Proportion of data to use to fit the model.

    References
    ----------
    .. [Calissano2021]  Calissano, A., Zeni, G., Fontana, M., Vantini, S.
        “Conformal Prediction Sets for Populations of Graphs.” Mox report 42, 2021.
        https://www.mate.polimi.it/biblioteca/add/qmox/42-2021.pdf
    """

    def __init__(self, space, metric, predrule, alpha=0.1, calibration_size=0.7):
        self.space = space
        self.metric = metric
        self.predrule = predrule
        self.alpha = alpha
        self.calibration_size = calibration_size

    def fit(self, dataset):
        """Class for Computing Conformal Prediction Sets using prediction rule.

        Implementing conformal prediction sets using conformal prediction with amplitude
        modulation described in [Calissano2022].

        Parameters
        ----------
        dataset : PointSet or array-like, shape=[..., n_nodes, n_nodes]
            Graphset for which we want to compute the interval.

        Return
        ------
        conformal_extremes : array-like, shape=[2, n_nodes, n_nodes]
            Two graphs representing the extremes of the set.

        References
        ----------
        .. [Calissano2021]  Calissano, A., Zeni, G., Fontana, M., Vantini, S.
            “Conformal Prediction Sets for Populations of Graphs.” Mox report 42, 2021.
            https://www.mate.polimi.it/biblioteca/add/qmox/42-2021.pdf
        """
        n = dataset.shape[1]
        nt = int(gs.floor(n * self.calibration_size))
        idx = gs.random.permutation(n)
        idx_train, idx_cal = idx[:nt], idx[nt:]

        estimator = self.predrule.fit(dataset[idx_train])
        estimate = estimator.estimate_

        data_sd = self.predrule.aligned_X_.std(axis=0)

        data_dev = self.metric.align_point_to_point(
            base_graph=estimate, graph_to_permute=dataset[idx_cal]
        )

        epsilon = 0.00001
        res_norm = abs(data_dev - estimate) / (data_sd + epsilon)
        scores = res_norm.max(axis=1)
        err = gs.quantile(scores, 1 - self.alpha)

        return gs.array([estimate - err * data_sd, estimate + err * data_sd])
