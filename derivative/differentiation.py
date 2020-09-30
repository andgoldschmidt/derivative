import abc
import numpy as np


methods = {}
default = ["finite_difference", {"k": 1}]


def register(name=""):
    def inner(f):
        n = name or f.__name__
        methods[n] = f
        return f
    return inner


def dxdt(x, t, kind=None, axis=1, **kwargs):
    """
    Compute the derivative of x with respect to t along axis using the numerical derivative specified by "kind". This is
    the functional interface of the Derivative class.

    This function requires that X and t have equal length along axis. An empty X results in an empty derivative. If
    X.shape[axis] == 1, then the derivative cannot be computed in a reasonable way and X is returned.

    For all other inputs, the implementation 'kind' is responsbile for determining the behavior.

    Args:
        x (:obj:`ndarray` of float): Ordered measurement values.
        t (:obj:`ndarray` of float): Ordered measurement times.
        kind (string): Derivative method name.
            Built in kinds:
            - finite_difference. required kwargs: k (symmetric window size as index).
            - savitzky_golay. required kwargs: order (of a fit polynomial), left, right (window size).
            - spectral. required kwargs: None.
            - spline. required kwargs: s (smoothing).
            - trend_filtered. required kwargs: order (of a fit polynomial), alpha (regularization).
        axis ({0,1}). axis of x along which to differentiate. default 1.
        **kwargs: Keyword arguments for the derivative method "kind".

    Returns:
        :obj:`ndarray` of float: Returns dx/dt along axis.
    """
    if kind is None:
        method = methods.get(default[0])
        return method(**default[1]).d(x, t, axis=axis)
    else:
        method = methods.get(kind)
        return method(**kwargs).d(x, t, axis=axis)


class Derivative(abc.ABC):
    """ Interface for computing numerical derivatives. """

    @abc.abstractmethod
    def compute(self, t, x, i):
        """
        Compute the derivative of one-dimensional data x with respect to t at the index i of x, (dx/dt)[i].

        Computation of a derivative should fail explicitely if the implementation is unable to compute a derivative at
        the desired index. Used for global differentiation methods, for example.

        This requires that x and t have equal lengths >= 2, and that the index i is a valid index.

        For each implementation, any exceptions that can be raised from a valid input should either be handled or the
        exception should be denoted in the docstring of that implementation. For example, some implementations may raise
        an exception when x and t have length 1.

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

    def d(self, X, t, axis=1):
        """
        Compute the derivative of measurements X taken at times t.

        An empty X results in an empty derivative. If X.shape[axis] == 1, then the derivative cannot be computed in a
        reasonable way and X is returned.
        
        Args:
            t (:obj:`ndarray` of float): Ordered measurement times.
            X  (:obj:`ndarray` of float): Ordered measurements values. Multiple measurements allowed.
            axis ({0,1}). axis of X along which to differentiate. default 1.

        Returns:
            :obj:`ndarray` of float: Returns dX/dt along axis.

        Raises:
            ValueError: Requires that X.shape[axis] equals len(t). If X is flat, requires that len(X) equals len(t).
        """
        # Cast
        X = np.array(X)
        if not X.size:
            return np.array([])

        flat = False
        # Check shape and axis
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            flat = True
        elif len(X.shape) == 2:
            if axis == 0:
                X = X.T
            elif axis == 1:
                pass
            else:
                raise ValueError("Invalid axis.")
        else:
            raise ValueError("Invalid shape of X.")

        if X.shape[1] != len(t):
            raise ValueError("Desired X axis size does not match t size.")

        # Differentiate if 2 or more points along axis
        if X.shape[1] == 1:
            dX = X
        else:
            dX = np.array([list(self.compute_for(t, x, np.arange(len(t)))) for x in X])

        if flat:
            return dX.flatten()
        else:
            return dX if axis == 1 else dX.T
