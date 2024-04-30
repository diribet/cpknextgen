from dataclasses import dataclass, field

import numpy as np

from cpknextgen.transformation import Transformation
from cpknextgen.normal_mixture.parameters_estimate import VariationalMixture
from cpknextgen.capability_index import CapabilityIndex, ProcessStats
from cpknextgen.graphics import GraphicsResult, get_results_graphics
from cpknextgen.custom_exceptions import InvalidDataError, InvalidParameterError


def small_infinity_handling(point_process_low, point_process_center, point_process_high,
                            sample_process_low, sample_process_center, sample_process_high,
                            data_boundaries):
    coefficient = 1e-8
    # Getting rif of too many decimals
    sample_process_center = np.round(sample_process_center, 20)
    sample_process_low = np.round(sample_process_low, 20)
    sample_process_high = np.round(sample_process_high, 20)

    point_process_center = np.round(point_process_center, 20)
    point_process_low = np.round(point_process_low, 20)
    point_process_high = np.round(point_process_high, 20)

    # Handling same values for quantiles
    if data_boundaries[0] is None and data_boundaries[1] is None:
        sample_process_low[sample_process_low == sample_process_center] -= np.abs(
            sample_process_low[sample_process_low == sample_process_center]) * coefficient + coefficient
        sample_process_high[sample_process_high == sample_process_center] += np.abs(
            sample_process_high[sample_process_high == sample_process_center]) * coefficient + coefficient

        point_process_low[point_process_low == point_process_center] -= np.abs(
            point_process_low[point_process_low == point_process_center]) * coefficient + coefficient
        point_process_high[point_process_high == point_process_center] += np.abs(
            point_process_high[point_process_high == point_process_center]) * coefficient + coefficient
    elif data_boundaries[1] is None:
        sample_process_high[sample_process_high == sample_process_center] += np.abs(
            sample_process_high[sample_process_high == sample_process_center]) * coefficient + coefficient
        sample_process_center[sample_process_low == sample_process_center] += np.abs(
            sample_process_center[sample_process_low == sample_process_center]) * coefficient + coefficient
        sample_process_high[sample_process_high == sample_process_center] += np.abs(
            sample_process_high[sample_process_high == sample_process_center]) * coefficient + coefficient

        point_process_high[point_process_high == point_process_center] += np.abs(
            point_process_high[point_process_high == point_process_center]) * coefficient + coefficient
        point_process_center[point_process_low == point_process_center] += np.abs(
            point_process_center[point_process_low == point_process_center]) * coefficient + coefficient
        point_process_high[point_process_high == point_process_center] += np.abs(
            point_process_high[point_process_high == point_process_center]) * coefficient + coefficient
    elif data_boundaries[0] is None:
        sample_process_low[sample_process_low == sample_process_center] -= np.abs(
            sample_process_low[sample_process_low == sample_process_center]) * coefficient + coefficient
        sample_process_center[sample_process_high == sample_process_center] -= np.abs(
            sample_process_center[sample_process_high == sample_process_center]) * coefficient + coefficient
        sample_process_low[sample_process_low == sample_process_center] -= np.abs(
            sample_process_low[sample_process_low == sample_process_center]) * coefficient + coefficient

        point_process_low[point_process_low == point_process_center] -= np.abs(
            point_process_low[point_process_low == point_process_center]) * coefficient + coefficient
        point_process_center[point_process_high == point_process_center] -= np.abs(
            point_process_center[point_process_high == point_process_center]) * coefficient + coefficient
        point_process_low[point_process_low == point_process_center] -= np.abs(
            point_process_low[point_process_low == point_process_center]) * coefficient + coefficient
    else:
        if np.sum(sample_process_low == sample_process_high) > 0:
            sample_process_low[sample_process_low == sample_process_high] -= np.minimum(
                np.abs(sample_process_low[sample_process_low == sample_process_high]) * coefficient + coefficient,
                np.abs(data_boundaries[0] - sample_process_low[sample_process_low == sample_process_high]) / 2)
            sample_process_high[sample_process_low == sample_process_high] += np.minimum(
                np.abs(sample_process_high[sample_process_low == sample_process_high]) * coefficient + coefficient,
                np.abs(data_boundaries[1] - sample_process_high[sample_process_low == sample_process_high]) / 2)
        sample_process_center[sample_process_low == sample_process_center] += np.minimum(
            np.abs(sample_process_center[sample_process_low == sample_process_center]) * coefficient + coefficient, np.abs(
                sample_process_high[sample_process_low == sample_process_center] - sample_process_center[
                    sample_process_low == sample_process_center]) / 2)
        sample_process_center[sample_process_high == sample_process_center] -= np.minimum(
            np.abs(sample_process_center[sample_process_high == sample_process_center]) * coefficient + coefficient, np.abs(
                sample_process_center[sample_process_high == sample_process_center] - sample_process_low[
                    sample_process_high == sample_process_center]) / 2)

        if np.sum(point_process_low == point_process_high) > 0:
            point_process_low[point_process_low == point_process_high] -= np.minimum(
                np.abs(point_process_low[point_process_low == point_process_high]) * coefficient + coefficient,
                np.abs(data_boundaries[0] - point_process_low[point_process_low == point_process_high]) / 2)
            point_process_high[point_process_low == point_process_high] += np.minimum(
                np.abs(point_process_high[point_process_low == point_process_high]) * coefficient + coefficient,
                np.abs(data_boundaries[1] - point_process_high[point_process_low == point_process_high]) / 2)
        point_process_center[point_process_low == point_process_center] += np.minimum(
            np.abs(point_process_center[point_process_low == point_process_center]) * coefficient + coefficient, np.abs(
                point_process_high[point_process_low == point_process_center] - point_process_center[
                    point_process_low == point_process_center]) / 2)
        point_process_center[point_process_high == point_process_center] -= np.minimum(
            np.abs(point_process_center[point_process_high == point_process_center]) * coefficient + coefficient, np.abs(
                point_process_center[point_process_high == point_process_center] - point_process_low[
                    point_process_high == point_process_center]) / 2)

    return sample_process_low, sample_process_center, sample_process_high, point_process_low, point_process_center, point_process_high


@dataclass
class EvaluationResult:
    """
    Container for results of function `evaluate`
    `standard_computation`: Bool indicating if the standard computation was performed. Nonstandard computation is
    performed when computed upper specification limit is lower than specified lower specification limit or other way
    around.
    `boundary_occurrences`: Tuple of floats indicating probability of occurrence of data at the extreme boundaries.
    alpha_prior: float, prior value of alpha parameter
    beta_prior: float, prior value of beta parameter
    m0_prior: float, prior value of m0 parameter
    v_prior: float, prior value of v parameter
    W0_prior: float, prior value of W0 parameter
    kernels: float, number of kernels in the mixture model which are activly used
    jittered_data: np.ndarray, jittered data used in the computation of the mixture model (if no jittering was
    performed, the original data is returned)
    """
    capability_index: CapabilityIndex = field(default=CapabilityIndex())
    graphics: GraphicsResult = field(default_factory=GraphicsResult)
    standard_computation: bool = field(default=True)
    boundary_occurrences: list = field(default_factory=lambda: [])

    alpha_prior: float = field(default=None)
    beta_prior: float = field(default=None)
    m0_prior: float = field(default=None)
    v_prior: float = field(default=None)
    W0_prior: float = field(default=None)

    kernels: float = field(default=None)
    jittered_data: np.ndarray = field(default=None)


def evaluate(
        data,
        specification_limits: tuple = (None, None),
        data_boundaries: tuple = (None, None),
        limit_quantiles: tuple = (0.01, 0.99),
        samples: float = 200,
        weight_concentration_prior: float = None,
        degrees_of_freedom_prior: float = None,
        covariance_prior: float = None,
        mean_precision_prior: float = None,
        mean_prior: float = None,
        responsibilities_option: str = "quantile",
        jitter_kernel: str = 'real_uniform',
        max_kernels_option: str = 'flexible',
        confidence: float = 0.95,
        resolution: float = None,
        random_seed: int = None):
    """
    ## Parameters:

    `data`: One-dimensional list or numpy.ndarray of measured values

    `specification_limits`: Tuple containing the 2 limits, lower limit first

    `data_boundaries`: Tuple containing 2 numbers such that no data point is found beyond them, lower limit first

    `limit_quantiles`: If no `specification_limits` are given, they are estimated as given quantiles of data,
                             by default (0.01, 0.99). Such estimated limits are used in transformations,
                             but not for capability index calculation.

    `samples`: Number of samples of mixtures to generate, by default set to 200


    `weight_concentration_prior`: By default set to 1/200. The higher the concentration parameter, the more components
                                  are open.

    `degrees_of_freedom_prior`: By default set to 1/200. The higher the value, the more confident we are with the
                                initial parameters.

    `covariance_prior`: By default set to 200. Lower values will result in wider components.

    `mean_precision_prior`: By default set to 1/200

    `mean_prior`: By default set to the mean of the transformed data is used as the prior mean.

    `responsibilities_option`: By default, it is set to 'quantile', meaning responsibilities are initialized based on
                                distance from quantiles of the data. Other options are 'random' (randomly initialized
                                responsibilities), 'random_from_data' (randomly chosen components from data points),
                                'kmeanplusplus' (k-means++ algorithm) and 'k_mean' (the initialized
                                responsibilities are calculated using the k-means algorithm).
                                Options 'random', 'random_from_data', 'kmeanplusplus' and 'k_mean' are taken from
                                the `sklearn.mixture.BayesianGaussianMixture` class.

    `jitter_kernel`: By default set to 'real_uniform' (non-random uniform). Other options are 'no_jitter' and 'uniform'
                    (random uniform).

    `max_kernels_option`: By default, this is set to 'flexible', which means that the number of initialized kernels
                          is equal to the minimum of the number of unique values within the range of the extreme bounds
                          minus one and 20. An alternative option is 'restricted', which sets the number of
                          kernels to 20.

    `confidence`: By default set to 0.95 to calculate 95% confidence interval for the capability indexes estimates.

    `resolution`: Resolution of measurement device. If not given, it is estimated as the smallest difference between two
                  unique values in the data divided by 2.

    ## Return

    Instance of class `EvaluationResult`

    """

    if type(data) is list:
        data = np.array(data)

    evaluation_result = EvaluationResult()

    _specification_limits = list(specification_limits)
    for limit_idx in [0, 1]:
        if all(map(lambda x: x is not None, [specification_limits[limit_idx], data_boundaries[limit_idx]])) and \
                specification_limits[limit_idx] == data_boundaries[limit_idx]:
            _specification_limits[limit_idx] = None

    if len(np.unique(data)) < 2:
        raise InvalidDataError("More unique values are required to calculate capability indexes.")

    if len(data) < 3:
        raise InvalidDataError("More data points are required to calculate capability indexes.")

    if responsibilities_option not in ['k_mean', 'random', 'random_from_data', 'kmeanplusplus', 'quantile']:
        raise InvalidParameterError("Invalid responsibilities option input.")

    if jitter_kernel not in ['no_jitter', 'real_uniform', 'uniform']:
        raise InvalidParameterError("Invalid jitter kernel input.")

    if max_kernels_option not in ['flexible', 'restricted']:
        raise InvalidParameterError("Invalid max kernels option input.")

    t = Transformation(data,
                       lower_specification_limit=_specification_limits[0],
                       upper_specification_limit=_specification_limits[1],
                       limit_quantiles=limit_quantiles,
                       lower_boundary=data_boundaries[0],
                       upper_boundary=data_boundaries[1],
                       jitter_kernel=jitter_kernel,
                       resolution=resolution
                       )

    if weight_concentration_prior is None:
        # bigger (>1) --> more kernels being active
        weight_concentration_prior = 1 / 200
    if degrees_of_freedom_prior is None:
        # if we're not confident that true covariance is similar to W0 --> lower value
        degrees_of_freedom_prior = 1 / 200
    if mean_precision_prior is None:
        # bigger (>1) --> cluster means are closer to `m0`
        mean_precision_prior = 1 / 200
    if covariance_prior is None:
        covariance_prior = 200
    if mean_prior is None:
        mean_prior = np.mean(t.data_transformed)

    standard_computation = t.standard_computation

    p1 = t.lower_boundary_data / t.len_of_data
    p2 = t.upper_boundary_data / t.len_of_data
    boundary_probabilities = np.array([p1, p2])
    boundary_data = [t.lower_boundary, t.upper_boundary]

    if max_kernels_option == "restricted":
        max_kernels = 20
    else:
        max_kernels = np.min([len(np.unique(t.data_transformed)) - 1, 20])

    vm = VariationalMixture(t.data_transformed.reshape(-1, 1),
                            max_kernels=max_kernels,
                            weight_concentration_prior=weight_concentration_prior,
                            mean_precision_prior=mean_precision_prior,
                            degrees_of_freedom_prior=degrees_of_freedom_prior,
                            covariance_prior=covariance_prior,
                            mean_prior=mean_prior,
                            r_option=responsibilities_option,
                            random_seed=random_seed,
                            boundary_probabilities=boundary_probabilities,
                            boundary_data=boundary_data
                            )

    vm.estimate_hyper_parameters()

    if random_seed is not None:
        vm.re_seed()

    mixtures = vm.sample(n_samples=samples)

    mixtures.estimate_quantiles()
    point_mixture = vm.point_estimate()
    point_mixture.estimate_quantiles()

    # if enough values are found at the extreme boundaries, use them instead of the quantile estimates
    if point_mixture.low_quantile_samples in boundary_data:
        point_process_low = point_mixture.low_quantile_samples
        sample_process_low = mixtures.low_quantile_samples
    else:
        point_process_low = t.complete_transformation_inverse(point_mixture.low_quantile_samples)
        sample_process_low = t.complete_transformation_inverse(mixtures.low_quantile_samples)

    if point_mixture.high_quantile_samples in boundary_data:
        point_process_high = point_mixture.high_quantile_samples
        sample_process_high = mixtures.high_quantile_samples
    else:
        point_process_high = t.complete_transformation_inverse(point_mixture.high_quantile_samples)
        sample_process_high = t.complete_transformation_inverse(mixtures.high_quantile_samples)

    if point_mixture.median_samples in boundary_data:
        point_process_center = point_mixture.median_samples
        sample_process_center = mixtures.median_samples
    else:
        point_process_center = t.complete_transformation_inverse(point_mixture.median_samples)
        sample_process_center = t.complete_transformation_inverse(mixtures.median_samples)

    ProcessStats.confidence = confidence

    sample_process_low, sample_process_center, sample_process_high, point_process_low, point_process_center, \
        point_process_high = small_infinity_handling(point_process_low, point_process_center, point_process_high,
                                                     sample_process_low, sample_process_center, sample_process_high,
                                                     data_boundaries)

    evaluation_result.capability_index = CapabilityIndex(
        process_low=ProcessStats(sample=sample_process_low, point=point_process_low),
        process_high=ProcessStats(sample=sample_process_high, point=point_process_high),
        process_center=ProcessStats(sample=sample_process_center, point=point_process_center),
        specification_lower=_specification_limits[0],
        specification_upper=_specification_limits[1],
        lower_boundary=data_boundaries[0],
        upper_boundary=data_boundaries[1])

    evaluation_result.graphics = get_results_graphics(data, t, mixtures, point_mixture, boundary_probabilities,
                                                      t.upper_boundary, t.lower_boundary, confidence=confidence)
    evaluation_result.boundary_occurrences = boundary_probabilities

    evaluation_result.alpha_prior = vm.alpha0
    evaluation_result.beta_prior = vm.beta0
    evaluation_result.means_prior = vm.m0
    evaluation_result.v_prior = vm.degrees_of_freedom_prior
    evaluation_result.W_prior = vm.W0

    evaluation_result.kernels = vm.num_kernels
    evaluation_result.jittered_data = t.data

    evaluation_result.standard_computation = standard_computation

    return evaluation_result
