from .differentiation import Derivative
import numpy as np


class FiniteDifference(Derivative):
    def __init__(self, k, **kwargs):
        """
        Compute the symmetric numerical derivative of equally-spaced data using the Taylor series. Derivatives at the
        boundaries are computed with the available reduction of the window or first order finite difference.

        Args:
            k (int): Interpolate the data with a polynomial through the 2k+1 points x[i-k], ..., x[i], ... x[i+k]
            **kwargs: Optional keyword arguments.

        Keyword Args:
            periodic (bool): If True, wrap the data. If False, compute edges separately. Default False.
        """
        self.k = k
        self.periodic = kwargs.get('periodic', False)

    @staticmethod
    def _symfd(i, x, dt, k):
        res = []
        # Construct each coefficient from the Taylor series.
        coefficient = 1
        for j in range(1, k + 1):
            coefficient *= (k - j + 1) / (k + j)
            alpha_j = 2 * np.power(-1, j + 1) * coefficient
            res.append(alpha_j * (x[(i + j) % len(x)] - x[(i - j) % len(x)]) / (2 * j) / dt)
        return np.sum(res)

    def compute(self, t, x, i):
        dt = t[1] - t[0]

        if not self.periodic:
            # Check for boundaries.
            if i == 0:
                return (x[i+1] - x[i])/dt
            elif i == len(x) - 1:
                return (x[i] - x[i-1])/dt

            # Check for overflow.
            left = i - self.k
            right = (len(x) - 1) - (i + self.k)
            overflow = min(left, right)
            if overflow < 0:
                return self._symfd(i, x, dt, self.k + overflow)

        # Default behavior (periodic data, symmetric derivative)
        return self._symfd(i, x, dt, self.k)


class SavitzkyGolay(Derivative):
    def __init__(self, order, left, right, **kwargs):
        """ Compute the numerical derivative by first finding the best  (least-squares) polynomial of order m < 2k
        using the k points in the neighborhood [t-left, t+right]. The derivative is computed  from the coefficients of
        the polynomial.

        Args:
            left: left edge of the window is t-left
            right: right edge of the window is t+right
            order:  order of polynomial (m < points in window)
        """
        # Note: Left and right have units (they do not count points)
        self.left = left
        self.right = right
        self.order = order

    def compute(self, t, x, i):
        i_l = np.argmin(np.abs(t - (t[i] - self.left)))
        i_r = np.argmin(np.abs(t - (t[i] + self.right)))

        # window too sparse. TODO: issue warning.
        if self.order > (i_r - i_l):
            return np.nan

        # Construct polynomial in t and do least squares regression
        # TODO! Make this robust!
        try:
            polyn_t = np.array([np.power(t[i_l:i_r], n)
                                for n in range(self.order + 1)]).T
            w, _, _, _ = np.linalg.lstsq(polyn_t, x[i_l:i_r], rcond=None)
        except np.linalg.LinAlgError:
            # Failed to converge, return bad derivative
            return np.nan

        # Compute derivative from fit
        return np.sum([j * w[j] * np.power(t[i], j - 1)
                       for j in range(1, self.order + 1)])

    def compute_for(self, t, x, indices):
        # If the window cannot reach any points, throw an exception
        # (likely the user forgets to rescale the window parameter)
        if min(t[1:] - t[:-1]) > max(self.left, self.right):
            raise ValueError("Found bad window ({}, {}) for x-axis data."
                             .format(self.left, self.right))
        for d in super().compute_for(t, x, indices):
            yield d
