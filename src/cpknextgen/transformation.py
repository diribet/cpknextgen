import scipy.stats
import numpy as np
from scipy import optimize

from cpknextgen.custom_exceptions import InvalidDataError


def jitter_data(data, lower_boundary=None, upper_boundary=None, kernel="no_jitter", resolution=None):
    if resolution is None:
        differences = np.diff(np.sort(data))
        differences_without_zeros = differences[differences != 0]
        resolution = np.min(differences_without_zeros)

    resolution_half = resolution / 2

    if lower_boundary is None:
        lower_boundary_data = data - resolution_half
    else:
        lower_boundary_data = np.maximum(data - resolution_half, lower_boundary)

    if upper_boundary is None:
        upper_boundary_data = data + resolution_half
    else:
        upper_boundary_data = np.minimum(data + resolution_half, upper_boundary)

    if kernel == "uniform":
        jittered_data = np.random.uniform(lower_boundary_data, upper_boundary_data)
        return jittered_data

    if kernel == "real_uniform":
        jittered_data = np.zeros(len(data))
        unique_data = np.unique(data)
        counter = 0
        for i in range(len(unique_data)):
            n = np.sum(data == unique_data[i])

            if lower_boundary is None:
                a = unique_data[i] - resolution_half
            else:
                a = np.maximum(unique_data[i] - resolution_half, lower_boundary)

            if upper_boundary is None:
                b = unique_data[i] + resolution_half
            else:
                b = np.minimum(unique_data[i] + resolution_half, upper_boundary)

            for j in range(n):
                jittered_data[counter] = a + (b - a) / (n + 1) * (j + 1)
                counter += 1
        return jittered_data

    return data


class YeoJohnson:
    def __init__(self, data):
        if type(data) is list:
            data = np.array(data)
        self.lmbda = self._optimize(data)
        self.transformed = self.transform(data, self.lmbda)

    def _optimize(self, x) -> np.float64:
        """
            Source: https://github.com/scikit-learn/scikit-learn/blob/f3f51f9b611bf873bd5836748647221480071a87/sklearn/preprocessing/_data.py#L3233
            Using this because scipy doesn't check for invalid values in np.log
        """
        x_tiny = np.finfo(np.float64).tiny

        def _neg_log_likelihood(lmbda):
            """Return the negative log likelihood of the observed data x as a
            function of lambda."""

            x_transformed = self.transform(x, lmbda)
            n_samples = x.shape[0]
            x_transfsormed_var = x_transformed.var()

            # Reject transformed data that would raise a RuntimeWarning in np.log
            if x_transfsormed_var < x_tiny:
                return np.inf

            log_var = np.log(x_transfsormed_var)
            loglike = -n_samples / 2 * log_var
            loglike += (lmbda - 1) * (np.sign(x) * np.log1p(np.abs(x))).sum()

            return -loglike

        # the computation of lambda is influenced by NaNs so we need to
        # get rid of them
        x = x[~np.isnan(x)]
        # choosing bracket 0, 2 so we do not have a problem with the inverse transformation
        lmbda = np.float64(optimize.fminbound(_neg_log_likelihood, x1=0, x2=2, full_output=False))
        if lmbda < 0.01:
            lmbda = 0
        if lmbda > 1.99:
            lmbda = 2
        return lmbda

    def transform(self, data, lmbda=None):
        if lmbda is None:
            lmbda = self.lmbda
        with np.errstate(all='raise'):
            try:
                transformed = scipy.stats.yeojohnson(data, lmbda)
            except Exception as e:
                self.lmbda = 1  # identity transform
                transformed = data
        return transformed

    def derivative_transform(self, x):
        """Return transformed input x following Yeo-Johnson transform with
        parameter lambda.
        """
        out = np.zeros_like(x)
        pos = x >= 0  # binary mask

        # when x >= 0
        if abs(self.lmbda) < np.spacing(1.):
            out[pos] = 1 / (x[pos] + 1)
        else:  # lmbda != 0
            out[pos] = np.power(x[pos] + 1, self.lmbda - 1)

        # when x < 0
        if abs(self.lmbda - 2) > np.spacing(1.):
            if 1 - self.lmbda < 0:
                out[~pos] = 1 / np.power(1 - x[~pos], -(1 - self.lmbda))
            else:
                out[~pos] = np.power(1 - x[~pos], 1 - self.lmbda)
        else:  # lmbda == 2
            out[~pos] = 1 / (1 - x[~pos])

        return out

    def inverse(self, x):
        max_bound = 30
        out = np.zeros_like(x)
        where_nonzero = x >= 0

        # when x >= 0
        if abs(self.lmbda) < np.spacing(1.0):
            x[where_nonzero] = np.minimum(x[where_nonzero], max_bound)
            out[where_nonzero] = np.exp(x[where_nonzero]) - 1
        else:  # lmbda != 0
            x[where_nonzero] = np.minimum(x[where_nonzero], np.exp(max_bound*self.lmbda)/self.lmbda)
            out[where_nonzero] = np.power((x[where_nonzero] * self.lmbda + 1), 1 / self.lmbda) - 1

        # when x < 0:
        if abs(self.lmbda - 2) > np.spacing(1.0):
            x[~where_nonzero] = np.maximum(x[~where_nonzero], - np.exp(max_bound * (2 - self.lmbda)) / (2 - self.lmbda))
            out[~where_nonzero] = 1 - np.power(-(2 - self.lmbda) * x[~where_nonzero] + 1, 1 / (2 - self.lmbda))
        else:  # lmbda == 2
            x[~where_nonzero] = np.maximum(x[~where_nonzero], -max_bound)
            out[~where_nonzero] = 1 - np.exp(-x[~where_nonzero])

        return out


class Transformation:
    def __init__(self, data: np.ndarray,
                 lower_specification_limit: float = None,
                 upper_specification_limit: float = None,
                 limit_quantiles: tuple = (),
                 lower_boundary: float = None,
                 upper_boundary: float = None,
                 jitter_kernel: str = 'real_uniform',
                 resolution: float = None
                 ):

        if type(data) is not np.ndarray:
            data = np.array(data)

        self.data = jitter_data(data, lower_boundary, upper_boundary, kernel=jitter_kernel, resolution=resolution)

        self.standard_computation = True
        self.lower_boundary = lower_boundary
        self.upper_boundary = upper_boundary

        self.len_of_data = len(self.data)

        self.upper_boundary_data = np.sum(self.data == self.upper_boundary)
        self.lower_boundary_data = np.sum(self.data == self.lower_boundary)

        self.boundaryless_data = self.data[self.data != self.upper_boundary]
        self.boundaryless_data = self.boundaryless_data[self.boundaryless_data != self.lower_boundary]

        if len(np.unique(self.boundaryless_data)) < 3 and jitter_kernel == 'no_jitter':
            raise InvalidDataError("More unique values are required to calculate capability indexes.")

        self.normalization_limits, self.standard_computation = self._check_limits(
            self.boundaryless_data,
            lower_specification_limit,
            upper_specification_limit,
            limit_quantiles,
            lower_boundary,
            upper_boundary)

        self.limits_to_boundaries = self.boundaries_transformation(self.normalization_limits)
        data_to_boundaries = self.boundaries_transformation(self.boundaryless_data)

        limits_to_normalisation_1 = np.array([-1, 1])
        data_normalisation_1 = self.normalisation_transformation(data_to_boundaries, order=1)

        self.yeojohnson = YeoJohnson(data_normalisation_1)

        self.limits_to_yeo = self.yeojohnson.transform(limits_to_normalisation_1)

        self.data_transformed = self.complete_transformation(self.boundaryless_data)

    @staticmethod
    def _check_limits(data, lower_specification_limit: float = None,
                      upper_specification_limit: float = None,
                      limit_quantiles: tuple = (),
                      lower_boundary: float = None,
                      upper_boundary: float = None) -> tuple:
        standard_computation = True
        if upper_boundary is not None and upper_boundary < np.max(data):
            raise InvalidDataError("User supplied upper boundary is less than maximum of data!")
        if lower_boundary is not None and lower_boundary > np.min(data):
            raise InvalidDataError("User supplied lower boundary is greater than minimum of data!")
        if len(limit_quantiles) != 2:
            limit_quantiles = (0.01, 0.99)

        lower_normalization_limit = np.quantile(data, limit_quantiles[
            0], method="inverted_cdf") if lower_specification_limit is None else lower_specification_limit
        upper_normalization_limit = np.quantile(data, limit_quantiles[
            1], method="inverted_cdf") if upper_specification_limit is None else upper_specification_limit

        if lower_normalization_limit >= upper_normalization_limit:
            standard_computation = False
            if len(np.unique(data)) > 1:
                if lower_specification_limit is None:
                    if lower_boundary is None:
                        lower_normalization_limit = (np.min([lower_normalization_limit, upper_normalization_limit]) -
                                                     np.min(np.diff(np.sort(np.unique(data)))))
                    else:
                        lower_normalization_limit = np.max([np.min([lower_normalization_limit, upper_normalization_limit]) -
                                                            np.min(np.diff(np.sort(np.unique(data)))), lower_boundary])
                if upper_specification_limit is None:
                    if upper_boundary is None:
                        upper_normalization_limit = (np.max([lower_normalization_limit, upper_normalization_limit]) +
                                                     np.min(np.diff(np.sort(np.unique(data)))))
                    else:
                        upper_normalization_limit = np.min([np.max([lower_normalization_limit, upper_normalization_limit]) +
                                                            np.min(np.diff(np.sort(np.unique(data)))), upper_boundary])

        return np.array([lower_normalization_limit, upper_normalization_limit]), standard_computation

    def boundaries_transformation(self, data, inverse=False):
        # maximum value to avoid overflow
        absolute_bound = 30
        if self.lower_boundary is None and self.upper_boundary is None:
            return data

        elif self.lower_boundary is None:
            if not inverse:
                return -np.log(- (data - self.upper_boundary))
            else:
                data = np.maximum(data, -absolute_bound)
                return -np.exp(-data) + self.upper_boundary

        elif self.upper_boundary is None:
            if not inverse:
                return np.log(data - self.lower_boundary)
            else:
                data = np.minimum(data, absolute_bound)
                return np.exp(data) + self.lower_boundary
        else:
            if not inverse:
                scaled_data = (data - self.lower_boundary) / (self.upper_boundary - self.lower_boundary)
                return np.log(scaled_data / (1 - scaled_data))
            else:
                data = np.minimum(data, absolute_bound)
                k = self.upper_boundary - self.lower_boundary
                return self.lower_boundary + (np.exp(data) / (1 + np.exp(data)) * k)

    def boundaries_transformation_derivative(self, data):
        if self.lower_boundary is None and self.upper_boundary is None:
            return np.ones_like(data)

        elif self.lower_boundary is None:
            return -1 / (data - self.upper_boundary)

        elif self.upper_boundary is None:
            return 1 / (data - self.lower_boundary)
        else:
            k = self.upper_boundary - self.lower_boundary
            return k / (data - self.lower_boundary) / (self.upper_boundary - data)

    def normalisation_transformation(self, data, inverse=False, order=2):
        if order == 1:
            normalization_limits = self.limits_to_boundaries
        else:
            normalization_limits = self.limits_to_yeo

        if normalization_limits[0] is None or normalization_limits[1] is None:
            return data

        _diff = normalization_limits[1] - normalization_limits[0]
        if not inverse:
            return (2 * (data - normalization_limits[0]) / _diff) - 1
        else:
            return (data + 1) / 2 * _diff + normalization_limits[0]

    def normalisation_transformation_derivative(self, data, order=2):
        if order == 1:
            normalization_limits = self.limits_to_boundaries
        else:
            normalization_limits = self.limits_to_yeo

        if normalization_limits[0] is None or normalization_limits[1] is None:
            return np.ones_like(data)
        _diff = normalization_limits[1] - normalization_limits[0]
        return np.ones_like(data) * 2 / _diff

    def complete_transformation(self, data):
        data_to_boundaries = self.boundaries_transformation(data)
        data_normalisation_1 = self.normalisation_transformation(data_to_boundaries, order=1)

        data_to_yeo = self.yeojohnson.transform(data_normalisation_1)
        data_normalisation_2 = self.normalisation_transformation(data_to_yeo)
        return data_normalisation_2

    def complete_transformation_inverse(self, data, order=4):
        data_normalisation_2_inverse = self.normalisation_transformation(data, inverse=True)
        if order == 1:
            return data_normalisation_2_inverse
        data_to_yeo_inverse = self.yeojohnson.inverse(data_normalisation_2_inverse)
        if order == 2:
            return data_to_yeo_inverse
        data_normalisation_1_inverse = self.normalisation_transformation(data_to_yeo_inverse, inverse=True, order=1)
        if order == 3:
            return data_normalisation_1_inverse
        data_to_boundaries_inverse = self.boundaries_transformation(data_normalisation_1_inverse, inverse=True)
        return data_to_boundaries_inverse
