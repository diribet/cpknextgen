import numpy as np
from dataclasses import dataclass, field

from cpknextgen.transformation import Transformation
from cpknextgen.normal_mixture.inference import MixtureDistributions

from scipy.stats import norm


def mixture_of_normals_density(weights, means, scales, x_grid):
    samples_len, len_of_means = means.shape
    samples = np.zeros(shape=x_grid.shape)
    for j in range(len_of_means):
        temp_grid = x_grid
        temp_grid = (temp_grid - means[0, j]) / scales[0, j]
        temp_grid[np.where(temp_grid > 20)] = 20
        temp_grid[np.where(temp_grid < -20)] = -20
        if j == 0:
            samples = weights[0, j] * norm.pdf(temp_grid, loc=0, scale=1) / scales[0, j]
        else:
            samples += weights[0, j] * norm.pdf(temp_grid, loc=0, scale=1) / scales[0, j]
    return samples


@dataclass
class GraphicsResult:
    empirical_cdf: dict = field(default_factory=dict)
    median_sample_cdf: dict = field(default_factory=dict)
    point_cdf: dict = field(default_factory=dict)
    lower_sample_cdf: dict = field(default_factory=dict)
    upper_sample_cdf: dict = field(default_factory=dict)
    point_density: dict = field(default_factory=dict)


def get_results_graphics(raw_measured_data,
                         transformation: Transformation,
                         mixtures: MixtureDistributions,
                         point_mixture: MixtureDistributions,
                         boundary_probabilities: np.ndarray,
                         upper_boundary: float,
                         lower_boundary: float,
                         confidence: float):
    """
    Returns the following graphics on the original axis:
    1. empirical CDF of the raw measured data (x: data_ordered, y: dataa_z_scores)
    2. lower and upper symmetrical bound on the ecdf by Dvoretzky–Kiefer–Wolfowitz inequality
     (x: data_ordered, y: :data_z_score_lower), (x: data_ordered, y: :data_z_score_upper)
    3. lower, upper symmetrical bound with 50% quantile of the mixture model CDF (x: lower_sample_cdf_to_plot,
    y: grid_to_plot), (x: upper_sample_cdf_to_plot, y: grid_to_plot), (x: median_sample_cdf_to_plot, y: grid_to_plot)
    4. point estimate of the density of the mixture model (x: original_axis_x, y: rescales_density_values)
    """
    data_ordered = np.sort(raw_measured_data)
    empirical_cdf_epsilon = np.sqrt(np.log(2 / (1 - confidence)) / (2 * len(raw_measured_data)))
    y_data_ecdf = np.linspace(0, 1, len(raw_measured_data) + 1, endpoint=False)[1:]
    data_z_scores = norm.ppf(y_data_ecdf)
    data_z_score_upper = norm.ppf(np.minimum(y_data_ecdf + empirical_cdf_epsilon, 1))
    data_z_score_lower = norm.ppf(np.maximum(y_data_ecdf - empirical_cdf_epsilon, 0))

    graphics_result = GraphicsResult()
    # Empirical CDF with upper and lower bounds of the confidence interval
    graphics_result.empirical_cdf = dict(x=data_ordered, y=data_z_scores)
    graphics_result.upper_empirical_cdf = dict(x=data_ordered, y=data_z_score_upper)
    graphics_result.lower_empirical_cdf = dict(x=data_ordered, y=data_z_score_lower)

    # np.nanquantile ignores NaN
    mixture_model_cdf_samples_x_low = np.nanquantile(mixtures.percentile_samples, (1 - confidence) / 2, axis=0)
    mixture_model_cdf_samples_x_median = np.nanquantile(mixtures.percentile_samples, 0.5, axis=0)
    mixture_model_cdf_samples_x_high = np.nanquantile(mixtures.percentile_samples, confidence + (1 - confidence) / 2,
                                                      axis=0)

    with np.errstate(invalid="ignore"):  # ignore "invalid value" warnings; result array can contain NaN
        ecdf_lower_error = transformation.complete_transformation_inverse(mixture_model_cdf_samples_x_low)
        median_sample_cdf = transformation.complete_transformation_inverse(mixture_model_cdf_samples_x_median)
        ecdf_upper_error = transformation.complete_transformation_inverse(mixture_model_cdf_samples_x_high)
        ecdf_point_error = transformation.complete_transformation_inverse(point_mixture.percentile_samples)

    grid_to_plot = mixtures.z_grid.copy()
    median_sample_cdf_to_plot = median_sample_cdf.copy()
    ecdf_lower_error_to_plot = ecdf_lower_error.copy()
    ecdf_upper_error_to_plot = ecdf_upper_error.copy()
    ecdf_point_error_to_plot = ecdf_point_error.copy().reshape(-1)

    if boundary_probabilities[0] is not None and boundary_probabilities[1] is not None:
        grid_to_plot_quantile = norm.cdf(grid_to_plot)
        if boundary_probabilities[0] != 0:
            grid_to_plot_quantile = np.insert(grid_to_plot_quantile, 0, 0)
            median_sample_cdf_to_plot = np.insert(median_sample_cdf_to_plot, 0, lower_boundary)
            ecdf_lower_error_to_plot = np.insert(ecdf_lower_error_to_plot, 0, lower_boundary)
            ecdf_upper_error_to_plot = np.insert(ecdf_upper_error_to_plot, 0, lower_boundary)
            ecdf_point_error_to_plot = np.insert(ecdf_point_error_to_plot, 0, lower_boundary)
        if boundary_probabilities[1] != 0:
            grid_to_plot_quantile = np.append(grid_to_plot_quantile, 1)
            median_sample_cdf_to_plot = np.append(median_sample_cdf_to_plot, upper_boundary)
            ecdf_lower_error_to_plot = np.append(ecdf_lower_error_to_plot, upper_boundary)
            ecdf_upper_error_to_plot = np.append(ecdf_upper_error_to_plot, upper_boundary)
            ecdf_point_error_to_plot = np.append(ecdf_point_error_to_plot, upper_boundary)
        grid_to_plot_quantile = grid_to_plot_quantile * (1 - boundary_probabilities[1] - boundary_probabilities[0]) + \
                                boundary_probabilities[0]
        grid_to_plot = norm.ppf(grid_to_plot_quantile)

    graphics_result.median_sample_cdf = dict(x=median_sample_cdf_to_plot, y=grid_to_plot)
    graphics_result.lower_sample_cdf = dict(x=ecdf_lower_error_to_plot, y=grid_to_plot)
    graphics_result.upper_sample_cdf = dict(x=ecdf_upper_error_to_plot, y=grid_to_plot)
    graphics_result.point_cdf = dict(x=ecdf_point_error_to_plot, y=grid_to_plot)

    # density calculation
    lower_bound_estimated = np.min(median_sample_cdf_to_plot)
    upper_bound_estimated = np.max(median_sample_cdf_to_plot)

    how_many_points = 202
    if transformation.upper_boundary is not None and transformation.lower_boundary is not None:
        y = transformation.complete_transformation(
            np.linspace(transformation.lower_boundary, transformation.upper_boundary,
                        how_many_points)[1:-1])
    elif transformation.upper_boundary is not None:
        y = transformation.complete_transformation(np.linspace(lower_bound_estimated,
                                                               transformation.upper_boundary, how_many_points)[1:-1])
    elif transformation.lower_boundary is not None:
        y = transformation.complete_transformation(
            np.linspace(transformation.lower_boundary, upper_bound_estimated,
                        how_many_points)[1:-1])
    else:
        y = transformation.complete_transformation(
            np.linspace(lower_bound_estimated, upper_bound_estimated, how_many_points)[1:-1])

    result1 = mixture_of_normals_density(point_mixture.weights_samples, point_mixture.means_samples,
                                         point_mixture.scales_samples, y)

    y2 = transformation.complete_transformation_inverse(y, order=1)
    result2 = transformation.normalisation_transformation_derivative(y2)

    y3 = transformation.complete_transformation_inverse(y, order=2)
    result3 = transformation.yeojohnson.derivative_transform(y3)

    y4 = transformation.complete_transformation_inverse(y, order=3)
    result4 = transformation.normalisation_transformation_derivative(y4, order=1)

    original_axis_x = transformation.complete_transformation_inverse(y, order=4)
    result5 = transformation.boundaries_transformation_derivative(original_axis_x)

    density_y = result1 * (result2 * result3 * result4 * result5)

    # When some datapoints are on extreme boundaries, the density needs to be rescaled
    if boundary_probabilities[0] is not None and boundary_probabilities[1] is not None:
        normalising_coefficient = 1 - boundary_probabilities[0] - boundary_probabilities[1]
    elif boundary_probabilities[0] is not None:
        normalising_coefficient = 1 - boundary_probabilities[0]
    elif boundary_probabilities[1] is not None:
        normalising_coefficient = 1 - boundary_probabilities[1]
    else:
        normalising_coefficient = 1

    rescales_density_values = density_y * normalising_coefficient
    graphics_result.point_density = dict(x=original_axis_x, y=rescales_density_values)

    return graphics_result
