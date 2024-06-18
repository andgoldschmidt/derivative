# Test the implementation of the interface for each method
import derivative
from derivative import dxdt, methods, utils
import pytest
import numpy as np
import inspect

def test_register():
    # Check that every class is registered in methods
    class_list = inspect.getmembers(derivative.dlocal, inspect.isclass)\
                 + inspect.getmembers(derivative.dglobal, inspect.isclass)
    impl_names = [derivative.dlocal.__name__, derivative.dglobal.__name__]
    must_register = [m[0] for m in class_list if m[1].__module__ in impl_names]

    # Interface is registered in each, ignore
    assert len(methods) == len(must_register)


def test_axis():
    """
    Checking the default method only. The other implementations follow if they satisfy the interface contract.
    """
    ts = np.arange(10)
    xs = np.array([ts, ts**2])
    shape1 = xs.shape
    # Regular: 2,10
    test1 = dxdt(xs, ts, axis=1).shape
    assert shape1 == test1
    # Transpose: 10,2
    test2 = dxdt(xs.T, ts, axis=0).shape[::-1]
    assert shape1 == test2
    # Flat
    shape2 = ts.shape
    test3 = dxdt(ts, ts, axis=1).shape
    assert shape2 == test3
    n = 4
    x3d = np.tile(np.arange(n), (n, n, 1))
    test_4 = dxdt(x3d, np.arange(n), axis=2)
    np.testing.assert_array_equal(test_4, np.ones((n,n,n)))


def test_empty():
    empty = np.array([])
    # finite_difference
    assert 0 == dxdt(empty, empty, kind='finite_difference', k=1).size
    # savitzky_golay
    assert 0 == dxdt(empty, empty, kind='savitzky_golay', order=1, left=2, right=2, iwindow=True).size


def test_one():
    one = np.arange(1)
    twobyone = np.arange(2).reshape(2,1)
    for data in [one, twobyone]:
        # spectral
        assert np.all(data == dxdt(data, one, kind='spectral'))
        # spline
        assert np.all(data == dxdt(data, one, kind='spline', order=1, s=.01))
        # trend_filtered
        assert np.all(data == dxdt(data, one, kind='trend_filtered', order=1, alpha=.01, max_iter=1e3))
        # finite_difference
        assert np.all(data == dxdt(data, one, kind='finite_difference', k=1))
        # savitzky_golay
        assert np.all(data == dxdt(data, one, kind='savitzky_golay', order=1, left=2, right=2, iwindow=True))


def test_small():
    two = np.arange(2)
    three = np.arange(3)

    # spectral - No errors
    for data in [two, three]:
        assert dxdt(data, data, kind='spectral').shape == data.shape

    # spline - TypeError: length of input > order for spline interpolation
    kwargs = {'s': .01, 'order': 2}
    with pytest.raises(TypeError):
        dxdt(two, two, kind='spline', **kwargs)
    assert three.shape == dxdt(three, three, kind='spline', **kwargs).shape

    # trend_filtered - ValueError: Requires that the number of points n > order + 1 to compute the objective
    kwargs = {'order': 1, 'alpha': .01, 'max_iter': int(1e3)}
    with pytest.raises(ValueError):
        dxdt(two, two, kind='trend_filtered', **kwargs)
    assert three.shape == dxdt(three, three, kind='trend_filtered', **kwargs).shape

    # finite_difference - No errors
    for data in [two, three]:
        assert data.shape == dxdt(data, data, kind='finite_difference', k=1).shape

    # savitzky_golay - RankWarning: The fit may be poorly conditioned if order >= points in window
    kwargs = {'left': 2, 'right': 2, 'iwindow': True, 'order': 2}
    with pytest.warns(UserWarning): # numpy.RankWarning is of type UserWarning
        assert two.shape == dxdt(two, two, kind='savitzky_golay', **kwargs).shape
    assert three.shape == dxdt(three, three, kind='savitzky_golay', **kwargs).shape


def test_hyperparam_entrypoint():
    func = utils._load_hyperparam_func("kalman.default")
    expected = 1
    result = func(None, None)
    assert result == expected


def test_negative_axis():
    t = np.arange(3)
    x = np.random.random(size=(2, 3, 2))
    x[1, :, 1] = 1
    axis = -2
    expected = np.zeros(3)
    dx = dxdt(x, t, kind='finite_difference', axis=axis, k=1)
    assert x.shape == dx.shape
    np.testing.assert_array_almost_equal(dx[1, :, 1], expected)