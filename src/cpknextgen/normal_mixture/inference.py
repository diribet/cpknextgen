import numpy as np
import scipy.stats as st
from numba import njit

from cpknextgen.custom_exceptions import InvalidDataError


@njit(cache=True)
def calculate_quantiles(n_samples, cdf_space, cdf_values, weights_array, means_array, scales_array, z_grid_cdf):
    """
    Function is not part of a class, because numba doesn't know how to work with classes.

    Calculate quantiles of interest for each sample mixture. The idea of the algorithm is to use the pre-calculated
    cdf values of the standard normal distribution and use linear interpolation to find quantiles, i.e. the
    inverse of the cdf.

    :param n_samples: int, number of mixtures to calculate the quantiles from
    :param cdf_space: numpy array, analyzed space for calculating CDF values of normal distribution with zero mean and
                      unit variance
    :param cdf_values: numpy array, calculated cdf values from the variable cdf_space
    :param weights_array: numpy array, array of weights, shape (n_samples, effective number of kernels)
    :param means_array: numpy array, array of means, shape (n_samples, effective number of kernels)
    :param scales_array: numpy array, array of scales, shape (n_samples, effective number of kernels)
    :param z_grid_cdf: numpy array, calculate the quantiles corresponding to the number of standard deviations for the
    normal distribution located in this array

    :return: samples: numpy array, corresponding quantiles of mixture to z_grid parameter
    """
    samples = np.zeros(shape=(n_samples, z_grid_cdf.shape[0]))
    len_of_xx_space = 10000

    # grid of quantiles to calculate
    len_of_grid = len(z_grid_cdf)
    len_of_means = len(means_array[0])

    for w in range(n_samples):
        weights = weights_array[w]
        means = means_array[w]
        scales = scales_array[w]

        z = [-3, 0, 3]
        start1 = sum(means * weights) + 2 * min(z + [-1]) * sum(scales ** 2) ** 0.5
        finish1 = sum(means * weights) + 2 * max(z + [1]) * sum(scales ** 2) ** 0.5

        start2 = np.min(means) - 4 * np.max(scales)
        finish2 = np.max(means) + 4 * np.max(scales)

        start = min([start1, start2])
        finish = max([finish1, finish2])

        # space, where quantiles can be found
        xx = np.linspace(start, finish, len_of_xx_space)

        value_cdf = np.array([0.])
        for i in range(len_of_means):
            value_cdf = value_cdf + np.interp((xx - means[i]) / scales[i], cdf_space, cdf_values) * weights[i]

        index_list = 0
        for i in range(len_of_grid):
            distance = np.abs(z_grid_cdf[i] - value_cdf[index_list])
            if index_list + 2 < len_of_xx_space:
                distance_next = np.abs(z_grid_cdf[i] - value_cdf[index_list + 1])
            else:
                # Constant used for termination since no two values can be further apart than 1
                distance_next = 2

            while distance >= distance_next:
                index_list += +1
                distance = np.abs(z_grid_cdf[i] - value_cdf[index_list])
                if index_list + 2 < len_of_xx_space:
                    distance_next = np.abs(z_grid_cdf[i] - value_cdf[index_list + 1])
                else:
                    distance_next = 2

            samples[w, i] = xx[index_list]

    return samples


class MixtureDistributions:
    """
    Class representing Gaussian Mixture Model initiated by either N samples of the model parameters or
    by their point estimates
    """

    def __init__(self, num_kernels: int,
                 n_samples: int,
                 weights_samples: np.ndarray,
                 means_samples: np.ndarray,
                 scales_samples: np.ndarray,
                 boundaries_data: list,
                 boundaries: list
                 ):
        """
        :param num_kernels: number of active components
        :param weights_samples,
        :param means_samples,
        :param scales_samples:
        the mixture parameters, shape of these arrays is (n_samples, num_kernels)
        if point estimates, n_samples = 1
        """

        self.num_kernels = num_kernels
        self.n_samples = n_samples
        self.weights_samples, self.means_samples, self.scales_samples = self._check_params_shape(
            weights_samples, means_samples, scales_samples)

        self.low_quantile_samples = np.zeros(self.n_samples)
        self.median_samples = np.zeros(self.n_samples)
        self.high_quantile_samples = np.zeros(self.n_samples)

        _interesting_z = range(-4, 5, 1)
        # concat linspaces that start at each interesting z so the resulting z_grid contains those z points exactly
        self.z_grid = np.concatenate(tuple(
            np.linspace(_interesting_z[idx - 1], _interesting_z[idx], num=13, endpoint=idx == 8) for idx in range(1, 9)
        ))

        self.boundaries = boundaries
        self.p1 = boundaries_data[0]
        self.p2 = boundaries_data[1]

        self.density_parts = []
        self.quantiles_of_interest = []

        # recalculates quantiles if there are data on extremes
        for i in [-3, 0, 3]:
            temp_result = (st.norm.cdf(i) - self.p1) / (1 - self.p1 - self.p2)
            temp_result_q = np.round(st.norm.ppf(temp_result),6)
            self.quantiles_of_interest.append(temp_result_q)
            if temp_result_q == temp_result_q:
                if temp_result_q not in self.z_grid:
                    self.z_grid = np.append(self.z_grid, [temp_result_q])

            current_quantile = st.norm.cdf(i)
            if current_quantile < self.p1:
                self.density_parts.append("Lower")
            elif current_quantile > 1 - self.p2:
                self.density_parts.append("Upper")
            else:
                self.density_parts.append("Middle")

        self.z_grid = np.sort(self.z_grid)

        self.percentile_samples = np.zeros(shape=(self.n_samples, self.z_grid.shape[0]))

        self.estimate_quantiles()

    def _check_params_shape(self, weights, means, scales):
        required_shape = (self.n_samples, self.num_kernels)
        to_array = lambda e: np.array(e) if type(e) is not np.ndarray else e
        weights = to_array(weights)
        means = to_array(means)
        scales = to_array(scales)
        try:
            assert weights.shape == required_shape
            assert means.shape == required_shape
            assert scales.shape == required_shape
        except AssertionError:
            raise InvalidDataError("""MixtureDistributions class expects weights, means and scales to be
             numpy arrays of shape (n_samples, num_kernels) where 
             n_samples ... # of samples of the mixture 
             num_kernels ... # of normal kernels in the mixture
             """)
        return weights, means, scales

    def estimate_quantiles(self):
        """
        Calculate quantiles of interest for each sample mixture.
        :return: None
        """
        cdf_space_preparation = np.linspace(-5, 5, 10000)
        cdf_values = st.norm.cdf(cdf_space_preparation)
        grid_cdf = st.norm.cdf(self.z_grid)

        samples = calculate_quantiles(self.n_samples, cdf_space_preparation, cdf_values, self.weights_samples,
                                      self.means_samples,
                                      self.scales_samples, grid_cdf)

        three_important_quantiles = []
        # if there are enough data on extremes, use them as quantiles, otherwise use the calculated quantiles from
        # the mixture model
        for i in range(3):
            if self.quantiles_of_interest[i] == self.quantiles_of_interest[i]:
                index_of_quantile = np.where(self.z_grid == self.quantiles_of_interest[i])[0][0]
                three_important_quantiles.append(samples[:, index_of_quantile])
            else:
                if self.density_parts[i] == "Lower":
                    three_important_quantiles.append(np.ones(self.n_samples) * self.boundaries[0])
                else:
                    three_important_quantiles.append(np.ones(self.n_samples) * self.boundaries[1])

        self.low_quantile_samples = three_important_quantiles[0]
        self.median_samples = three_important_quantiles[1]
        self.high_quantile_samples = three_important_quantiles[2]
        self.percentile_samples = samples
