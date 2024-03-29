# Run tests on utils.py methods
from derivative.utils import deriv, integ, _memoize_arrays
import pytest
import numpy as np


def test_deriv_bad():
    with pytest.raises(ValueError):
        deriv(1, 1, 1)

    with pytest.raises(ValueError):
        deriv(1, 1, 2)

    with pytest.raises(ValueError):
        deriv(3, 1, 3)


def test_integ_bad():
    with pytest.raises(ValueError):
        integ(0, 1)


def test_deriv_min():
    result = deriv(2, 1, 1)
    expect = np.array([[-1.,  1.]])
    assert np.all(np.isclose(result, expect))


def test_deriv_dx():
    result = deriv(2, .01, 1)
    expect = np.array([[-1.,  1.]])/.01
    assert np.all(np.isclose(result, expect))


def test_deriv_orders():
    e1 = np.array([[-1., 1., 0., 0., 0.],
                   [0., -1., 1., 0., 0.],
                   [0., 0., -1., 1., 0.],
                   [0., 0., 0., -1., 1.]])
    e2 = np.array([[1., -2., 1., 0., 0.],
                   [0., 1., -2., 1., 0.],
                   [0., 0., 1., -2., 1.]])
    e3 = np.array([[-1.,  3., -3.,  1.,  0.],
                   [ 0., -1.,  3., -3.,  1.]])
    e4 = np.array([[ 1., -4.,  6., -4.,  1.]])
    for i,e in enumerate([e1,e2,e3,e4]):
        result = deriv(5, 1, i+1)
        assert np.all(np.isclose(result, e))


def test_integ_min():
    result = integ(1,1)
    assert result.size == 0

    result = integ(2, 1)
    expect = np.array([[0. , 0. ],
                       [0.5, 0.5]])
    assert np.all(np.isclose(result, expect))


def test_integ_med():
    result = integ(5, 1)
    expect = np.array([[0. , 0. , 0. , 0. , 0. ],
                       [0.5, 0.5, 0. , 0. , 0. ],
                       [0.5, 1. , 0.5, 0. , 0. ],
                       [0.5, 1. , 1. , 0.5, 0. ],
                       [0.5, 1. , 1. , 1. , 0.5]])
    assert np.all(np.isclose(result, expect))


def test_integ_dx():
    result = integ(2, .01)
    expect = np.array([[0., 0.],
                       [0.005, 0.005]])
    assert np.all(np.isclose(result, expect))


def test_memoize_arrays():
    arr = np.array([1,2,3])
    @_memoize_arrays(maxsize=1)
    def sin(arr, addendum):
        return np.sin(arr) + addendum
    result = sin(arr, 1)
    sin(arr, 1)
    sin(2* arr, 1)
    assert sin.cache_info().hits == 1
    assert sin.cache_info().misses == 2
    assert sin.cache_info().currsize == 1
    np.testing.assert_almost_equal(result, np.sin(arr)+1)


def test_memoize_array_cachesize_zero():
    arr = np.array([1,2,3])

    @_memoize_arrays(maxsize=0)
    def sin(arr, addendum):
        return np.sin(arr) + addendum

    sin(arr, 1)
    sin(arr, 1)
    assert sin.cache_info().hits == 0
    assert sin.cache_info().misses == 2
    assert sin.cache_info().currsize == 0

def test_memoize_array_clear_cache():
    arr = np.array([1,2,3])

    @_memoize_arrays(maxsize=1)
    def sin(arr, addendum):
        return np.sin(arr) + addendum

    sin(arr, 1)
    sin.cache_clear()
    assert sin.cache_info().misses == 0
    assert sin.cache_info().currsize == 0

def test_memoize_array_kwargs():
    arr = np.array([1,2,3])

    @_memoize_arrays(maxsize=1)
    def sin(arr, addendum=1):
        return np.sin(arr) + addendum

    sin(arr, addendum=1)
    sin(arr, addendum=1)
    assert sin.cache_info().hits == 1
    assert sin.cache_info().misses == 1
    assert sin.cache_info().currsize == 1
