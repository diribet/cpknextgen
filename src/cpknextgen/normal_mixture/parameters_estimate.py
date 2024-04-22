"""
Dirichlet Process (=Infinite) Gaussian Mixture Model
- it's not Dirichlet Process but the vm algorithm still opens only necessary kernels
- the model is parametrized by weights w, means mu and scales S: Sum(k=1;K) = w_k * N(mu_k, S_k)
weights must sum up to 1

- the model parameters are random variables:
weights w ~ Dir(alpha)
means mu ~ N(m, inv(beta * L)), where L = inv(S)
precisions L ~ Wishart(W, v)

hence the hyper-parameters of the model are:
alpha ... weight concentration
m ... means
beta... mean precision
W ... covariance
v ... degrees of freedom

"""

import numpy as np
import scipy.stats as st

from numba import njit, vectorize

from sklearn.cluster import kmeans_plusplus, KMeans

from cpknextgen.normal_mixture.inference import MixtureDistributions
from cpknextgen.custom_exceptions import InvalidParameterError


@vectorize
def gammaln_nr(z):
    """Numerical Recipes 6.1"""
    taylor_coefficients = np.array([
        57.1562356658629235, -59.5979603554754912,
        14.1360979747417471, -0.491913816097620199,
        .339946499848118887e-4, .465236289270485756e-4,
        -.983744753048795646e-4, .158088703224912494e-3,
        -.210264441724104883e-3, .217439618115212643e-3,
        -.164318106536763890e-3, .844182239838527433e-4,
        -.261908384015814087e-4, .368991826595316234e-5])

    y = z
    temporary_variable = z + 5.24218750000000000
    temporary_variable = (z + 0.5) * np.log(temporary_variable) - temporary_variable
    series = 0.999999999999997092

    n = taylor_coefficients.shape[0]
    for j in range(n):
        y = y + 1.
        series = series + taylor_coefficients[j] / y

    return temporary_variable + np.log(2.5066282746310005 * series / z)


@vectorize
def py_digamma(x):
    """
    Function is not part of a class, because numba doesn't know how to work with classes.
    Calculates digamma value of x. We assume x > 0.

    :param x: numpy array

    :return: digamma value of x
    """
    x = np.array(x)
    r = 0
    while np.min(x) <= 5:
        r -= 1 / x
        x += 1
    f = 1 / (x * x)
    t = f * (-1 / 12.0 + f * (1 / 120.0 + f * (-1 / 252.0 + f * (1 / 240.0 + f * (-1 / 132.0 + f * (691 / 32760.0 +
        f * (-1 / 12.0 + f * 3617 / 8160.0)))))))
    return r + np.log(x) - 0.5 / x + t


@njit(cache=True)
def find_hyper_parameters(r, data, prior_alpha, prior_beta, prior_m, prior_W, prior_v, n_of_components=20,
                          max_iterations=2000):
    """
    Function is not part of a class, because numba doesn't know how to work with classes.
    Iterative algorithm that finds the parameters to be used to calculate the mean, standard deviation
    and weights of a mixture of normal distributions.

    Algorithm implemented from the book Pattern Recognition and Machine Learning by Christopher M. Bishop, ISBN-10 0387310738
    """
    prior_m = prior_m * np.ones((n_of_components, 1))
    prior_W = prior_W * np.ones((n_of_components, 1, 1))

    over_W_constant = 1 / prior_W
    n_of_data = len(data)
    padding = 10 ** (-40)
    accuracy = 0.0001

    component_size = r.sum(axis=0) + padding

    xm = np.empty(n_of_components)
    for i in range(n_of_components):
        xm[i] = np.sum(r[:, i] * data.T) / component_size[i]
    xm = xm.reshape((-1, 1))

    s_intermediate = np.empty(n_of_components)
    s_intermediate_updated = np.empty(n_of_components)
    for i in range(n_of_components):
        s_intermediate[i] = np.sum(r[:, i] * data.T) / component_size[i]
        s_intermediate_updated[i] = np.sum(r[:, i] * (data.T - s_intermediate[i]) ** 2) / component_size[i]
    s = s_intermediate_updated.reshape((-1, 1, 1))

    alpha = prior_alpha + component_size
    beta = prior_beta + component_size
    means = (prior_beta * prior_m + component_size.reshape(-1, 1) * xm) / beta.reshape(-1, 1)

    ein_sum_diag_new = np.square(prior_m[0] - s_intermediate)
    ein_sum_diag = ein_sum_diag_new.reshape(1, 1, n_of_components)

    W = 1 / (over_W_constant + (component_size * s.T).T + (prior_beta * component_size * ein_sum_diag / beta).T)
    v = prior_v + component_size

    log_constant = np.log(2)
    digamma_constant = py_digamma(np.sum(alpha))

    for j in range(max_iterations):
        prev_alpha = alpha
        prev_beta = beta
        prev_means = means
        prev_W = W
        prev_v = v
        # E STEP
        einsum = np.empty((n_of_data, n_of_components))
        for i in range(n_of_components):
            temp = (data - means[i]) ** 2 * W[i]
            einsum[:, i] = temp.reshape(1, -1)

        maha_sq = -0.5 * (1 / beta + v * einsum)

        ln_pi = py_digamma(alpha) - digamma_constant  # _estimate_log_weights
        ln_Lambda = py_digamma(0.5 * v) + log_constant + np.log(W[:, 0, 0])
        ln_p = ln_pi + 0.5 * ln_Lambda + maha_sq - 0.5 * np.log(2 * np.pi)
        ln_r = ln_p - np.log(padding + np.sum(np.exp(ln_p), axis=-1)).reshape(-1, 1)

        r = np.exp(ln_r)

        # M STEP
        component_size = r.sum(axis=0) + padding

        for i in range(n_of_components):
            xm[i] = np.sum(r[:, i] * data.T) / component_size[i]
            s_intermediate[i] = np.sum(r[:, i] * data.T) / component_size[i]
            s_intermediate_updated[i] = np.sum(r[:, i] * (data.T - s_intermediate[i]) ** 2) / component_size[i]
        xm = xm.reshape((-1, 1))
        s = s_intermediate_updated.reshape((-1, 1, 1))

        alpha = prior_alpha + component_size
        beta = prior_beta + component_size
        means = (prior_beta * prior_m + component_size.reshape(-1, 1) * xm) / beta.reshape(-1, 1)

        ein_sum_diag_new = np.square(prior_m[0] - s_intermediate)
        ein_sum_diag = ein_sum_diag_new.reshape(1, 1, n_of_components)

        W = 1 / (over_W_constant + (component_size * s.T).T + (prior_beta * component_size * ein_sum_diag / beta).T)
        v = prior_v + component_size

        stop = True

        if np.max(np.abs(prev_alpha / alpha - 1)) > accuracy:
            stop = False

        if stop is True:
            if np.max(np.abs(prev_beta / beta - 1)) > accuracy:
                stop = False
            if np.max(np.abs(prev_means / means - 1)) > accuracy:
                stop = False
            if np.max(np.abs(prev_v / v - 1)) > accuracy:
                stop = False
            if np.max(np.abs(prev_W / W - 1)) > accuracy:
                stop = False

        if stop:
            break
    return alpha, beta, means, W, v


class VariationalMixture:
    def __init__(self, data: np.ndarray,
                 max_kernels: int,
                 weight_concentration_prior: float = None,
                 mean_precision_prior: float = None,
                 degrees_of_freedom_prior: float = None,
                 covariance_prior: float = None,
                 mean_prior: float = None,
                 r_option: str = 'quantile',
                 random_seed: int = None,
                 boundary_probabilities=None,
                 boundary_data=None):

        self.r_option = r_option
        self.boundaries_data = boundary_probabilities
        self.boundaries = boundary_data
        # set random state if seed is given
        self.random_seed = random_seed
        if random_seed is not None:
            _bit_generator = np.random.PCG64(random_seed)
            self._rng = np.random.Generator(_bit_generator)
        else:
            self._rng = np.random.default_rng()

        # shape of data is (n_samples, n_dim)
        # for process data we have only one dimension
        self.data = data
        self.num_data_points = self.check_data_shape()
        # upper bound for the # of kernels
        # should be sufficiently larger than what we believe is needed
        self.max_kernels = max_kernels

        self.alpha0 = weight_concentration_prior
        self.beta0 = mean_precision_prior
        self.m0 = mean_prior
        self.degrees_of_freedom_prior = degrees_of_freedom_prior
        self.W0 = covariance_prior

        self.num_kernels = max_kernels
        self.has_hyperparams = False

        self.alpha = None
        self.beta = None
        self.means = None
        self.W = None
        self.degrees_of_freedom = None

    def check_data_shape(self):
        try:
            n_samples, n_dim = self.data.shape
            if n_dim > 1:
                raise ValueError("Shape of data must be (N, 1)")
            return n_samples

        except ValueError:
            raise Exception("Shape of data for Variational Mixture must be (N, 1)")

    def re_seed(self) -> None:
        self._rng = np.random.Generator(np.random.PCG64(self.random_seed))

    def estimate_hyper_parameters(self, max_iterations=2000) -> None:
        """
        Options "random", "k_mean", "random_from_data" and "kmeanplusplus" are from sklearn.
        Source: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/mixture/_base.py#L99
        Calculates initial responsibilities of the model and then estimates the hyper-parameters of the model.
        """
        r = None
        if self.r_option == 'random':
            r = np.random.uniform(size=(len(self.data), self.max_kernels))
            r /= r.sum(axis=1)[:, np.newaxis]

        if self.r_option == 'quantile':
            mean_quantiles = np.quantile(self.data, q=np.linspace(0, 1, self.max_kernels),
                                         method="interpolated_inverted_cdf")
            mean_quantiles = np.array(list(set(mean_quantiles)))
            r = np.zeros((len(self.data), len(mean_quantiles)))
            for i in range(len(self.data)):
                r[i, :] = np.abs(self.data[i] - mean_quantiles)
            r /= r.sum(axis=1)[:, np.newaxis]
            self.max_kernels = len(mean_quantiles)

        if self.r_option == 'k_mean':
            n_of_clusters = np.min([len(np.unique(self.data)) - 1, self.max_kernels])
            label = (
                KMeans(
                    n_clusters=n_of_clusters, n_init=1, random_state=0
                )
                .fit(self.data)
                .labels_
            )
            r = np.zeros((len(self.data), n_of_clusters))
            r[np.arange(len(self.data)), label] = 1
            r /= r.sum(axis=1)[:, np.newaxis]
            self.max_kernels = int(n_of_clusters)

        if self.r_option == "random_from_data":
            r = np.zeros((len(self.data), self.max_kernels))
            indices = np.random.choice(
                len(self.data), size=self.max_kernels, replace=False
            )
            r[indices, np.arange(self.max_kernels)] = 1

        if self.r_option == "kmeanplusplus":
            n_of_clusters = np.min([len(np.unique(self.data)) - 1, self.max_kernels])
            r = np.zeros((len(self.data), n_of_clusters))
            _, indices = kmeans_plusplus(
                self.data,
                n_of_clusters,
                random_state=0,
            )
            indices = np.array(indices)
            r[indices, np.arange(n_of_clusters)] = 1
            self.max_kernels = int(n_of_clusters)

        if r is None:
            raise InvalidParameterError("Invalid responsibilities option input.")

        alpha, beta, means, W, v = find_hyper_parameters(r, self.data, self.alpha0, self.beta0, self.m0,
                                                         self.W0, self.degrees_of_freedom_prior,
                                                         n_of_components=self.max_kernels,
                                                         max_iterations=max_iterations)

        sorting_indices_desc = np.argsort(alpha)[::-1]  # descending
        self.alpha = alpha[sorting_indices_desc]
        self.beta = beta[sorting_indices_desc]
        self.means = means[sorting_indices_desc]
        self.degrees_of_freedom = v[sorting_indices_desc]
        self.W = W[sorting_indices_desc]
        self.has_hyperparams = True

        self.num_kernels = (self.alpha != self.alpha0).sum()

    def point_estimate(self) -> MixtureDistributions:
        """
        Calculate point estimates of the model parameters (weights, means and scales)
        :return: instance of MixtureDistributions class; attributes are arrays of shapes: (1, num_kernels)
        """
        if not self.has_hyperparams:
            self.estimate_hyper_parameters()

        # expected value of Dirichlet distr.
        weights = self.alpha[:self.num_kernels] / self.alpha[:self.num_kernels].sum()
        # 1 / (expected value of Wishart distr.)
        covariances = 1 / (self.W[:self.num_kernels, 0, 0] * self.degrees_of_freedom[:self.num_kernels])
        return MixtureDistributions(
            self.num_kernels,
            1,
            weights[:self.num_kernels].reshape(1, self.num_kernels),
            self.means[:self.num_kernels].reshape(1, self.num_kernels),
            np.sqrt(covariances).reshape(1, self.num_kernels),
            boundaries_data=self.boundaries_data,
            boundaries=self.boundaries
        )

    def sample(self, n_samples=200, dilute=1) -> MixtureDistributions:
        """
        Simulate `n_samples` mixtures.
        The mixture parameters have the following prior distributions:
        w ~ Dir(alpha)
        mu ~ N(m, S), S = inv(beta * L)
        L ~ Wishart(W, v), inv(S) = L
        This method returns n_samples of w, mu, sqrt(S) drawn from corresponding distributions.
        :return: instance of MixtureDistributions class; attributes are arrays of shapes: (n_samples, num_kernels)
        """
        if not self.has_hyperparams:
            self.estimate_hyper_parameters()

        alpha = self.alpha[:self.num_kernels] * dilute
        beta = self.beta[:self.num_kernels] * dilute
        m = self.means[:self.num_kernels, :]
        W = self.W[:self.num_kernels, :, :] / dilute
        v = self.degrees_of_freedom[:self.num_kernels] * dilute

        # weights_samples, L_samples/S_samples, means_samples are matrices of n_samples rows & num_kernels columns
        # each row = one possible mixture model

        weights_samples = self._rng.dirichlet(alpha=alpha, size=n_samples)

        scale_samples = []

        L_samples = np.ones(shape=(n_samples, self.num_kernels))
        means_samples = np.ones(shape=(n_samples, self.num_kernels))
        for k in range(self.num_kernels):
            L_samples[:, k] = st.wishart(df=v[k], scale=W[k, 0, 0]).rvs(size=n_samples, random_state=self._rng)
            # for a given k, draw a random number from the following normal distributions
            # N(m[k], L_samples[0, k]), N(m[k], L_samples[1, k]), ..., N(m[k], L_samples[199, k]),
            scale = np.sqrt(1 / (beta[k] * L_samples[:, k]))
            means_samples[:, k] = self._rng.normal(loc=m[k], scale=scale)
            scale_samples.append(scale)

        return MixtureDistributions(self.num_kernels, n_samples, weights_samples, means_samples, np.sqrt(1 / L_samples),
                                    boundaries_data=self.boundaries_data, boundaries=self.boundaries)
