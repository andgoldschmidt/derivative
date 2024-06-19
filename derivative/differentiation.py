import abc
import numpy as np
from numpy.typing import NDArray

from .utils import _memoize_arrays

methods = {}
default = ["finite_difference", {"k": 1}]


def register(name=""):
    def inner(f):
        n = name or f.__name__
        methods[n] = f
        return f
    return inner


@_memoize_arrays()
def _gen_method(x, t, kind, axis, **kwargs):
    return methods.get(kind)(**kwargs)


def dxdt(x, t, kind=None, axis=1, **kwargs):
    """
    Compute the derivative of x with respect to t along axis using the numerical derivative specified by "kind". This is
    the functional interface of the Derivative class.

    This function requires that X and t have equal length along axis. An empty X results in an empty derivative. If
    X.shape[axis] == 1, then the derivative cannot be computed in a reasonable way and X is returned.

    The implementation 'kind', an instance of the Derivative class, is responsbile for determining the behavior.

    Args:
        x (:obj:`ndarray` of float): Ordered measurement values.
        t (:obj:`ndarray` of float): Ordered measurement times.
        kind (string): Derivative method name (see available kinds).
        axis ({0,1}): Axis of x along which to differentiate. Default 1.
        **kwargs: Keyword arguments for the derivative method "kind".

    Available kinds
        - finite_difference. Required kwargs: k (symmetric window size as index).
        - savitzky_golay. Required kwargs: order (of a fit polynomial), left, right (window size).
        - spectral. Required kwargs: None.
        - spline. Required kwargs: s (smoothing).
        - trend_filtered. Required kwargs: order (of a fit polynomial), alpha (regularization).

    Returns:
        :obj:`ndarray` of float: Returns dx/dt along axis.
    """
    if kind is None:
        method = _gen_method(x, t, default[0], axis, **default[1])
        return method.d(x, t, axis=axis)
    else:
        method = _gen_method(x, t, kind, axis, **kwargs)
        return method.d(x, t, axis=axis)


def smooth_x(x, t, kind=None, axis=1, **kwargs):
    """
    Compute the smoothed version of x given t along axis using the numerical
    derivative specified by "kind". This is the functional interface of
    the Derivative class's x() method.

    This function requires that X and t have equal length along axis. If
    X.shape[axis] == 1, then the smoothing cannot be computed in a reasonable way and X is returned.

    The implementation 'kind', an instance of the Derivative class, is responsbile for determining the behavior.

    Args:
        x (:obj:`ndarray` of float): Ordered measurement values.
        t (:obj:`ndarray` of float): Ordered measurement times.
        kind (string): Derivative method name (see available kinds).
        axis ({0,1}): Axis of x along which to differentiate. Default 1.
        **kwargs: Keyword arguments for the derivative method "kind".

    Available kinds
        - finite_difference. Required kwargs: k (symmetric window size as index).
        - savitzky_golay. Required kwargs: order (of a fit polynomial), left, right (window size).
        - spectral. Required kwargs: None.
        - spline. Required kwargs: s (smoothing).
        - trend_filtered. Required kwargs: order (of a fit polynomial), alpha (regularization).

    Returns:
        :obj:`ndarray` of float: Returns dx/dt along axis.
    """
    if kind is None:
        method = _gen_method(x, t, default[0], axis, **default[1])
        return method.x(x, t, axis=axis)
    else:
        method = _gen_method(x, t, kind, axis, **kwargs)
        return method.x(x, t, axis=axis)


class Derivative(abc.ABC):
    """ Interface for computing numerical derivatives. """

    @abc.abstractmethod
    def compute(self, t, x, i):
        """
        Compute the derivative of one-dimensional data x with respect to t at the index i of x, (dx/dt)[i].

        Computation of a derivative should fail explicitely if the implementation is unable to compute a derivative at
        the desired index. Used for global differentiation methods, for example.

        This requires that x and t have equal lengths >= 2, and that the index i is a valid index.

        For each implementation, any exceptions raised by a valid input should either be handled or denoted in the
        implementation docstring. For example, some implementations may raise an exception when x and t have length 2.

        Args:
            t (:obj:`ndarray` of float):  Ordered measurement times.
            x (:obj:`ndarray` of float):  Ordered measurement values.
            i (int): Index i at which to compute (dx/dt)[i]

        Returns:
            float: (dx/dt)[i]
        """

    def compute_for(self, t, x, indices):
        """
        Compute derivative (dx/dt)[i] for i in indices. Overload this if desiring a more efficient computation over a
        list of indices.

        This function requires that x and t have equal length along axis, and that all of the indicies are valid.

        Args:
            t (:obj:`ndarray` of float): Ordered measurement times.
            x (:obj:`ndarray` of float): Ordered measurement values.
            indices (:obj:`ndarray` of int): Indices i at which to compute (dx/dt)[i]

        Returns:
            Generator[float]: yields (dx/dt)[i] for i in indices
        """
        for i in indices:
            yield self.compute(t, x, i)

    def compute_x(self, t, x, i):
        """
        Compute smoothed values of one-dimensional data x at the index i of x.
        Overload this if subclass actually smooths values.

        This requires that x and t have equal lengths >= 2, and that the index i is a valid index.

        For each implementation, any exceptions raised by a valid input should either be handled or denoted in the
        implementation docstring. For example, some implementations may raise an exception when x and t have length 2.

        Args:
            t (:obj:`ndarray` of float):  Ordered measurement times.
            x (:obj:`ndarray` of float):  Ordered measurement values.
            i (int): Index i at which to returned smoothed values

        Returns:
            float
        """
        return x[i]

    def compute_x_for(self, t, x, indices):
        """
        Compute smoothed values of x at each i in indices. Overload
        this if desiring a more efficient computation over a list of
        indices.

        This function requires that x and t have equal length along axis, and that all of the indicies are valid.

        Args:
            t (:obj:`ndarray` of float): Ordered measurement times.
            x (:obj:`ndarray` of float): Ordered measurement values.
            indices (:obj:`ndarray` of int): Indices i at which to compute (dx/dt)[i]

        Returns:
            Generator[float]: yields (dx/dt)[i] for i in indices
        """
        for i in indices:
            yield self.compute_x(t, x, i)

    def d(self, X, t, axis=1):
        """
        Compute the derivative of measurements X taken at times t.

        An empty X results in an empty derivative. If X.shape[axis] == 1, then the derivative cannot be computed in a
        reasonable way and X is returned.
        
        Args:
            X  (:obj:`ndarray` of float): Ordered measurements values. Multiple measurements allowed.
            t (:obj:`ndarray` of float): Ordered measurement times.
            axis ({0,1}). axis of X along which to differentiate. default 1.

        Returns:
            :obj:`ndarray` of float: Returns dX/dt along axis.

        Raises:
            ValueError: Requires that X.shape[axis] equals len(t). If X is flat, requires that len(X) equals len(t).
        """
        X, flat = _align_axes(X, t, axis)

        if X.shape[1] == 1:
            dX = X
        else:
            dX = np.array([list(self.compute_for(t, x, np.arange(len(t)))) for x in X])

        return _restore_axes(dX, axis, flat)


    def x(self, X, t, axis=1):
        """
        Compute the smoothed X values from measurements X taken at times t.

        Not all methods perform smoothing when calculating derivatives.  In
        these cases, X is returned unmodified

        Args:
            X  (:obj:`ndarray` of float): Ordered measurements values. Multiple measurements allowed.
            t (:obj:`ndarray` of float): Ordered measurement times.
            axis ({0,1}). axis of X along which to smooth. default 1.

        Returns:
            :obj:`ndarray` of float: Returns dX/dt along axis.
        """
        X, orig_shape = _align_axes(X, t, axis)

        if X.shape[1] == 1:
            dX = X
        else:
            dX = np.array([list(self.compute_x_for(t, x, np.arange(len(t)))) for x in X])

        return _restore_axes(dX, axis, orig_shape)


def _align_axes(X, t, axis) -> tuple[NDArray, tuple[int, ...]]:
    X = np.array(X)
    orig_shape = X.shape
    # By convention, differentiate axis 1
    if len(orig_shape) == 1:
        X = X.reshape(1, -1)
    else:
        ax_len = orig_shape[axis]
        # order of operations coupled with _restore_axes.  Move differentiation axis to
        # zero so reshape does not skew differentiation axis
        X = np.moveaxis(X, axis, 0).reshape((ax_len, -1)).T
    if X.shape[1] != len(t):
        raise ValueError("Desired X axis size does not match t size.")
    return X, orig_shape


def _restore_axes(dX: NDArray, axis: int, orig_shape: tuple[int, ...]) -> NDArray:
    if len(orig_shape) == 1:
        return dX.flatten()
    else:
        # order of operations coupled with _align_axes
        orig_diff_axis = range(len(orig_shape))[axis] # to handle negative axis args
        extra_dims = tuple(
            length for ax, length in enumerate(orig_shape) if ax != orig_diff_axis
        )
        moved_shape = (orig_shape[axis],) + extra_dims
        dX = np.moveaxis(dX.T.reshape((moved_shape)), 0, axis)
        return dX
