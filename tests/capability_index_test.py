import pytest
import numpy as np
from utils import assert_almost_equal
import json
from cpknextgen import evaluate
from pathlib import Path


def load_test_data():
    test_data_path = Path(__file__).parent / "test_data" / "test_data.json"
    with open(test_data_path) as infile:
        return json.load(infile)


def run_data_test(file_index, test_data):
    # Retrieve the current test set from the test data
    current_set = test_data['file' + str(file_index)]

    # Extract the required data for the test
    data = np.array(current_set["data"])
    error = current_set["error"]
    upper_extreme = current_set["upper_extreme"]
    lower_extreme = current_set["lower_extreme"]
    upper_specification = current_set["upper_specification"]
    lower_specification = current_set["lower_specification"]

    if error is not None:
        with pytest.raises(Exception, match=error):
            evaluate(data,
                     specification_limits=(lower_specification, upper_specification),
                     data_boundaries=(lower_extreme, upper_extreme),
                     random_seed=0,
                     samples=200)

    else:
        cp_pointed_expected = current_set["cp_point"]
        cpk_pointed_expected = current_set["cpk_point"]
        cp_interval = current_set["cp_interval"]
        cpk_interval = current_set["cpk_interval"]

        # Perform the evaluation on the dataset
        result = evaluate(np.array(data),
                          specification_limits=(lower_specification, upper_specification),
                          data_boundaries=(lower_extreme, upper_extreme),
                          random_seed=0)

        # Assert the results
        assert_almost_equal(result.capability_index.cp.point, cp_pointed_expected, decimal=2)
        assert_almost_equal(result.capability_index.cpk.point, cpk_pointed_expected, decimal=2)
        for i in range(2):
            assert_almost_equal(result.capability_index.cp.interval[i], cp_interval[i], decimal=2)
            assert_almost_equal(result.capability_index.cpk.interval[i], cpk_interval[i], decimal=2)


test_data = load_test_data()


@pytest.mark.parametrize("file_index", [0, 40, 41, 64])
def test_index_extreme_bounds(file_index):
    # Test the capability index calculation of different data sets located on extreme bounds
    run_data_test(file_index, test_data)


@pytest.mark.parametrize("file_index", [1, 2, 3, 4, 5])
def test_index_continuous_data(file_index):
    # Test the capability index calculation of different continuous data sets
    run_data_test(file_index, test_data)


@pytest.mark.parametrize("file_index", [12, 14, 16, 18, 31])
def test_index_many_non_unique_values(file_index):
    # Test the capability index calculation of different continuous data sets
    run_data_test(file_index, test_data)


@pytest.mark.parametrize("file_index", [26, 47, 61, 72, 78])
def test_index_outliers(file_index):
    # Test the capability index calculation of different continuous data sets
    run_data_test(file_index, test_data)


@pytest.mark.parametrize("file_index", [29])
def test_index_heavy_tail(file_index):
    # Test the capability index calculation of different continuous data sets
    run_data_test(file_index, test_data)


@pytest.mark.parametrize("file_index", [8, 10, 178])
def test_error_non_unique_values(file_index):
    run_data_test(file_index, test_data)


@pytest.mark.parametrize("file_index", [559, 560, 561])
def test_error_not_enough_values(file_index):
    run_data_test(file_index, test_data)
