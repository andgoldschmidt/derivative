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
