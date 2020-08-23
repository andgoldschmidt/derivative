from derivative import dxdt, methods
import pytest
import numpy as np


_threshold = 1e-6


def default_args(kind):
    assert kind in methods

    if kind == 'spectral':
        return {}
    elif kind == 'spline':
        return {'s': .01}
    elif kind == 'trend_filtered':
        return {'order': 0, 'alpha': .01}
    elif kind == 'finite_difference':
        return {'k': 1}
    elif kind == 'savitzky_golay':
        return {'left': .5, 'right': .5, 'order': 3}
    else:
        raise ValueError('Unimplemented default args for kind {}.'.format(kind))


def constant_fn(kind):
    t = np.linspace(0,1,100)
    x = np.ones_like(t)
    return dxdt(x, t, kind=kind, **default_args(kind)), np.zeros_like(t)


def poly_fn(kind):
    t = np.linspace(0,1,100)
    x = t**2 + t + np.ones_like(t)
    return dxdt(x, t, kind=kind, **default_args(kind)), 2*t + np.ones_like(t)


def trig_fn(kind):
    t = np.linspace(0,1,100)
    x = np.sin(t)
    return dxdt(x, t, kind=kind, **default_args(kind)), np.cos(t)


def test_constant():
    for m in methods:
        test, ans = constant_fn(m)
        assert np.sum(test - ans)/len(test) < _threshold


def test_poly():
    for m in methods:
        test, ans = poly_fn(m)
        assert np.sum(test - ans)/len(test) < _threshold


def test_trig():
    for m in methods:
        test, ans = trig_fn(m)
        assert np.sum(test - ans)/len(test) < _threshold
