from functools import _CacheInfo as CacheInfo, wraps
from collections import OrderedDict, Counter
import numpy as np
from scipy.special import binom


def deriv(n, dx, order):
    """ Matrix derivative. Equi-spaced derivative via forward finite differences, mapping R^n into R^(n-order).

    Args:
        n (int): Number of data points
        dx (float): Width of x[k+1] - x[k]
        order (int): Order of finite difference method to use.

    Return:
        (:obj:`ndarray` of float): An n-by-(n-order) matrix derivative.

    Raises:
        ValueError: Requires n >= 2 and n - order >= 1.
    """
    if n < 2 or n - order < 1:
        raise ValueError('Bad size n={} for derivative of order {}'.format(n, order))

    # Implement finite difference at 1st order accuracy
    method = [(-1)**(order+i)*binom(order,i) for i in range(order+1)]
    # Construct matrix
    M = [method + [0] * (n - len(method))]
    for row in range(n - len(method)):
        M.append([0] * (row + 1) + method + [0] * (n - len(method) - row - 1))
    return np.array(M) / dx ** order


def integ(n, dx=1):
    """ Equi-spaced anti-derivative via the trapezoid rule mapping R^n to R^n.

    Args:
        n (int): Number of data points
        dx (float): Width of x[k+1] - x[k]

    Return:
        (:obj:`ndarray` of float): An n-by-n matrix integral.

    Raises:
        ValueError: Requires n in positive integers.
    """
    if n == 1:
        return np.array([])
    elif n == 2:
        return np.array([[0, 0],
                         [1, 1]]) * dx / 2
    elif n > 2:
        M = [[0] * n]
        for row in range(0, n - 1):
            M.append([1] + [2] * row + [1] + [0] * (n - row - 2))
        return np.array(M) * dx / 2
    else:
        raise ValueError('Bad size of {}'.format(n))


def _memoize_arrays(maxsize=128):
    """A cache wrapper for functions that accept numpy arrays.

    Cannot directly use any memoization from functools on these
    functions because they require hashable types.
    """
    def memoizing_decorator(wrapped_func):
        class ArrayKey(int):
            pass

        def make_key(*args, **kwargs):
            def arrs_to_keys(*args, **kwargs):
                new_args = []
                new_kwargs = {}
                for arg in args:
                    if not isinstance(arg, np.ndarray):
                        new_args.append(arg)
                        continue
                    key = ArrayKey(hash(arg.tobytes()))
                    new_args.append(key)
                for k, v in kwargs.items():
                    if not isinstance(v, np.ndarray):
                        new_kwargs[k] = v
                        continue
                    key = ArrayKey(hash(v.tobytes()))
                    new_kwargs[k] = key
                return new_args, new_kwargs
            new_args, new_kwargs = arrs_to_keys(*args, **kwargs)
            new_args_dict = {k: v for k, v in enumerate(new_args)}
            return (
                        tuple(sorted(new_args_dict.items()))
                        + tuple(sorted(new_kwargs.items()))
                    )

        arg_dict = OrderedDict()
        hits = 0
        misses = 0
        @wraps(wrapped_func)
        def wrapper_func(*args, **kwargs):
            nonlocal arg_dict, hits, misses
            cache_key = make_key(*args, **kwargs)
            try:
                result = arg_dict[cache_key]
                arg_dict.move_to_end(cache_key)
                hits += 1
                return result
            except KeyError: pass
            misses += 1
            result = wrapped_func(*args, **kwargs)
            if maxsize > 0:
                arg_dict[cache_key] = result
                if len(arg_dict) > maxsize:
                    arg_dict.popitem(last=False)
            return result

        def cache_clear():
            nonlocal arg_dict, hits, misses
            arg_dict = OrderedDict()
            hits = 0
            misses = 0
        wrapper_func.cache_clear = cache_clear

        def cache_info():
            return CacheInfo(hits, misses, maxsize, arg_dict.__len__())
        wrapper_func.cache_info = cache_info

        return wrapper_func

    return memoizing_decorator

