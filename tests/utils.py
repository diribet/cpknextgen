import numpy

# shadow numpy functions to handle None type


def assert_almost_equal(a, b, decimal=7):
    if all([a is None, b is None]):
        assert a == b
    else:
        numpy.testing.assert_almost_equal(a, b, decimal=decimal)


def assert_array_almost_equal(a, b, decimal=7):
    if all([k is None for k in a]) and all([k is None for k in b]):
        assert a == b
    else:
        numpy.testing.assert_array_almost_equal(a, b, decimal=decimal)


def assert_array_equal(a, b):
    if all([k is None for k in a]) and all([k is None for k in b]):
        assert a == b
    else:
        numpy.testing.assert_array_equal(a, b)


def assert_equal(a, b):
    if all([a is None, b is None]):
        assert a == b
    else:
        numpy.testing.assert_equal(a, b)
