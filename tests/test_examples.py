# Run tests that checks basic derivative examples. Use warnings to denote a mathematical failure.
import numpy as np
import pytest
from derivative import dxdt, smooth_x, methods
from derivative.differentiation import _gen_method


# Utilities for tests
# ===================
def default_args(kind):
    """ The assumption is that the function will have dt = 1/100 or 2pi/100 over an interval of length 1 or 2pi
    and not vary much. The goal is to to set the parameters such that we obtain effective derivatives under these
    conditions.
    """
    if kind == 'spectral':
        # frequencies should be guaranteed to be between 0 and 50 (a filter will reduce bad outliers)
        return {'filter': np.vectorize(lambda w: 1 if abs(w) < 10 else 0)}
    elif kind == 'spline':
        return {'s': 0.1}
    elif kind == 'trend_filtered':
        return {'order': 0, 'alpha': .01, 'max_iter': int(1e3)}
    elif kind == 'finite_difference':
        return {'k': 1}
    elif kind == 'savitzky_golay':
        return {'left': 5, 'right': 5, 'order': 3, 'iwindow': True}
    elif kind == 'kalman':
        return {'alpha': .05}
    elif kind == "kernel":
        return {"sigma": 1, "lmbd": .01, "kernel": "gaussian"}
    else:
        raise ValueError('Unimplemented default args for kind {}.'.format(kind))
    
class NumericalExperiment:
    def __init__(self, fn, fn_str, t, kind, args):
        self.fn = fn
        self.fn_str = fn_str
        self.t = t
        self.kind = kind
        self.kwargs = args
        self.axis = 1

    def run(self):
        return dxdt(self.fn(self.t), self.t, self.kind, self.axis, **self.kwargs)

def compare(experiment, truth, rel_tol, abs_tol, shape_only=False):
    """ Compare a numerical experiment to theoretical expectations. Issue warnings for derivative methods that fail,
    use asserts for implementation requirements.
    """
    values = experiment.run()
    message_main = "In {} dxdt applied to {}, ".format(experiment.kind, experiment.fn_str)
    message_sub = "output length of {} =/= expected length of {}.".format(len(values), len(truth))
    assert len(values) == len(truth), message_main + message_sub
    if len(values) > 0 and not shape_only:
        residual = (values-truth)/len(values)
        def mean_sq(x):
            return np.sqrt(np.sum(x ** 2 / x.size))
        assert mean_sq(residual) < max(abs_tol, mean_sq(truth) * rel_tol)
        # Median is robust to outliers
        assert np.median(np.abs(residual)) < max(abs_tol, np.median(np.abs(truth)) * rel_tol)
        # But make sure outliers are also looked at
        assert np.linalg.norm(residual, ord=np.inf) < max(abs_tol, np.linalg.norm(truth, ord=np.inf) * rel_tol)


# Check that only numbers are returned
# ====================================
@pytest.mark.parametrize("m", methods)
def test_notnan(m):
    t = np.linspace(0, 1, 100)
    nexp = NumericalExperiment(lambda t1: np.random.randn(*t1.shape), 'f(t) = t', t, m, default_args(m))
    values = nexp.run()
    message = "In {} dxdt applied to {}, np.nan returned instead of float.".format(nexp.kind, nexp.fn_str)
    assert not np.any(np.isnan(values)), message


# Test that basic functions are differentiated correctly
# ======================================================
funcs_and_derivs = (
    (lambda t: np.ones_like(t), "f(t) = 1", lambda t: np.zeros_like(t), "const1"),
    (lambda t: np.zeros_like(t), "f(t) = 0", lambda t: np.zeros_like(t), "const0"),
    (lambda t: t, "f(t) = t", lambda t: np.ones_like(t), "lin-identity"),
    (lambda t: 2 * t + 1, "f(t) = 2t+1", lambda t: 2 * np.ones_like(t), "lin-affine"),
    (lambda t: -t, "f(t) = -t", lambda t: -np.ones_like(t), "lin-neg"),
    (lambda t: t ** 2 - t + np.ones_like(t), "f(t) = t^2-t+1", lambda t: 2 * t -np.ones_like(t), "polynomial"),
    (lambda t: np.sin(t) + np.ones_like(t) / 2, "f(t) = sin(t)+1/2", lambda t: np.cos(t), "trig"),
    (lambda t: np.array([2 * t, - t]), "f(t) = [2t, -t]", lambda t: np.vstack((2 * np.ones_like(t), -np.ones_like(t))), "2D linear"),
    (lambda t: np.array([np.sin(t), np.cos(t)]), "f(t) = [sin(t), cos(t)]", lambda t: np.vstack((np.cos(t), -np.sin(t))),  "2D trig",
),
)
@pytest.mark.filterwarnings('ignore::sklearn.exceptions.ConvergenceWarning')
@pytest.mark.parametrize("m", methods)
@pytest.mark.parametrize("func_spec", funcs_and_derivs)
def test_fn(m, func_spec):
    func, fname, deriv, f_id = func_spec
    t = np.linspace(0, 2*np.pi, 100, endpoint=False) # For periodic functions, it's important the endpoint not be included
    if m == 'trend_filtered':
        # Add noise to avoid all zeros non-convergence warning for sklearn lasso
        f_mod = lambda t: func(t) + 1e-9 * np.random.randn(*t.shape) # rename to avoid infinite loop
    else:
        f_mod = func
    nexp = NumericalExperiment(f_mod, fname, t, m, default_args(m))
    bad_combo=False
    # spectral is only accurate for periodic data. Ideally fixed in decorators
    if ("lin" in f_id or "poly" in f_id) and m == "spectral":
        bad_combo=True
    compare(nexp, deriv(t), 1e-1, 1e-1, bad_combo)


# Test smoothing for those that do it
# ===================================
@pytest.mark.parametrize("kind", ("kalman", "trend_filtered"))
def test_smoothing_x(kind):
    t = np.linspace(0, 1, 100)
    rng = np.random.default_rng(10)
    x = np.sin(t) + rng.normal(scale=1e-1, size=t.shape)
    method = _gen_method(x, t, kind=kind, axis=1, **default_args(kind))
    x_est = method.x(x, t)
    # MSE
    assert np.linalg.norm(x_est - np.sin(t)) ** 2 / len(t) < 1e-1

@pytest.mark.parametrize("kind", ("kalman", "trend_filtered"))
def test_smoothing_functional(kind):
    t = np.linspace(0, 1, 100)
    rng = np.random.default_rng(10)
    x = np.sin(t) + rng.normal(scale=1e-1, size=t.shape)
    x_est = smooth_x(x, t, kind=kind, axis=1, **default_args(kind))
    # MSE
    assert np.linalg.norm(x_est - np.sin(t)) ** 2 / len(t) < 1e-1


# Test caching of the expensive _gen_method using a dummy
# =======================================================
@pytest.fixture
def clean_gen_method_cache():
    _gen_method.cache_clear()
    yield
    _gen_method.cache_clear()

def test_gen_method_caching(clean_gen_method_cache):
    x = np.ones(3)
    t = np.arange(3)
    expected = _gen_method(x, t, "finite_difference", 1, k=1)
    result = _gen_method(x, t, "finite_difference", 1, k=1)
    assert _gen_method.cache_info().hits == 1
    assert _gen_method.cache_info().misses == 1
    assert _gen_method.cache_info().currsize == 1
    assert id(expected) == id(result)

def test_gen_method_kwarg_caching(clean_gen_method_cache):
    x = np.ones(3)
    t = np.arange(3)
    expected = _gen_method(x, t, "finite_difference", 1, k=1)
    # different variants => cache misses
    result = _gen_method(x, t, "finite_difference", axis=1, k=1)
    _gen_method(x, t, kind="finite_difference", axis=1, k=1)
    assert _gen_method.cache_info().hits == 0
    assert _gen_method.cache_info().misses == 3
    assert _gen_method.cache_info().currsize == 3
    assert id(expected) != id(result)


# Test caching of the expensive private _global methods using a dummy
# ===================================================================
@pytest.fixture
def method_inst(request):
    x = np.ones(3)
    t = np.arange(3)
    method = _gen_method(x, t, request.param, 1, **default_args(request.param))
    method._global.cache_clear()
    yield x, t, method
    method._global.cache_clear()

@pytest.mark.filterwarnings('ignore::sklearn.exceptions.ConvergenceWarning')
@pytest.mark.parametrize("method_inst", ["kalman", "trend_filtered"], indirect=True)
def test_global_caching_xd(method_inst):
    # make sure we're not recomputing expensive _global() method
    x, t, method = method_inst
    method.x(x, t)
    method.d(x, t)
    assert method._global.cache_info().hits == 1
    assert method._global.cache_info().misses == 1
    assert method._global.cache_info().currsize == 1

@pytest.mark.filterwarnings('ignore::sklearn.exceptions.ConvergenceWarning')
@pytest.mark.parametrize("method_inst", ["kalman", "trend_filtered", "spectral"], indirect=True)
def test_global_caching_dd(method_inst):
    # make sure we're not recomputing expensive _global() method
    x, t, method = method_inst
    method.d(x, t)
    method.d(x, t)
    assert method._global.cache_info().hits == 1
    assert method._global.cache_info().misses == 1
    assert method._global.cache_info().currsize == 1

@pytest.mark.filterwarnings('ignore::sklearn.exceptions.ConvergenceWarning')
@pytest.mark.parametrize("method_inst", ["kalman", "trend_filtered"], indirect=True)
def test_cached_global_order(method_inst):
    x, t, method = method_inst
    x = np.vstack((x, -x))
    first_result = method.x(x, t, axis=1)
    second_result = method.x(x, t, axis=1)
    np.testing.assert_equal(first_result, second_result)
