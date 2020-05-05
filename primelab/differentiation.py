import abc
import numpy as np


class Derivative(abc.ABC):
    """ Interface for computing numerical derivatives. """

    @abc.abstractmethod
    def compute(self, t, x, i):
        """ Compute the derivative of x with respect to t at the index i of x, (dx/dt)[i].

        Computation of a derivative should fail explicitely if the implementation is unable to compute a derivative at
        the desired index. Used for global differentiation methods, for example.

        Args:
            t (:obj:`ndarray` of float):  Ordered measurement times.
            x (:obj:`ndarray` of float):  Ordered measurement values.
            i (int): Index i at which to compute (dx/dt)[i]

        Returns:
            float: (dx/dt)[i]
        """

    def compute_for(self, t, x, indices):
        """
        Compute derivative (dx/dt)[i] for i in indices. Overload this if
        desiring a more efficient computation over a list of indices.

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
        
        Args:
            t (:obj:`ndarray` of float): Ordered measurement times.
            X  (:obj:`ndarray` of float): Ordered measurement values.
            axis ({0,1}). axis of X along which to differentiate. default 1.

        Returns:
            :obj:`ndarray` of float: Returns dX/dt along axis.

        Raises:
            ValueError: Requires that X.shape[axis] equals len(t). If X is flat, requires that len(X) equals len(t).
        """
        # Cast
        X = np.array(X)
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

        # Differentiate
        dX = np.array([list(self.compute_for(t, x, np.arange(len(t)))) for x in X])
        if flat:
            return dX.flatten()
        else:
            return dX if axis == 1 else dX.T
