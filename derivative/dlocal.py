from .differentiation import Derivative, register
import numpy as np
from numpy.polynomial.polynomial import polyfit


@register("finite_difference")
class FiniteDifference(Derivative):
    def __init__(self, k, **kwargs):
        """
        Compute the symmetric numerical derivative of equally-spaced data using the Taylor series. Derivatives at the
        boundaries are computed with the available reduction of the window or first order finite difference.

        A simple first order finite difference scheme in the spirit of this code is available in numpy.gradient.

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


@register("savitzky_golay")
class SavitzkyGolay(Derivative):
    def __init__(self, order, left, right, **kwargs):
        """ Compute the numerical derivative by first finding the best (least-squares) polynomial of order m < 2k+1
        using the 2k+1 points in the neighborhood [t-left, t+right]. The derivative is computed  from the coefficients
        of the polynomial. The default edge behavior is to truncate the window at the offending edge.

        A simple symmetric version of the Savitzky-Galoy filter is available as scipy.signal.savgol_filter.

        Args:
            left (float): left edge of the window is t-left
            right (float): right edge of the window is t+right
            order (int):  order of polynomial (m < points in window)
            **kwargs: Optional keyword arguments.

        Keyword Args:
            use_iwindow (bool): Whether to use an indexed window. If True, left and right act as indicies for t instead
                of as lengths in units of t. Default False.
        """
        self.left = left
        self.right = right
        self.use_iwindow = kwargs.get('use_iwindow', False)
        self.order = order

    def _trunc_window(self, t, i):
        ileft = np.argmin(np.abs(t - (t[i] - self.left)))
        iright = np.argmin(np.abs(t - (t[i] + self.right)))
        return [ileft, iright]

    def compute(self, t, x, i):
        # Default edge behavior is to truncate the window.
        if self.use_iwindow:
            i_l, i_r = max(0., i - self.left), min(i + self.right, len(t) - 1)
        else:
            i_l, i_r = self._trunc_window(t, i)

        # Construct a polynomial in t using least squares regression.
        tfit = t[i_l:i_r]
        xfit = x[i_l:i_r]
        # Can raise RankWarning if order exceeds points in the window.
        w = polyfit(tfit, xfit, self.order)

        # Compute the derivative from the polyfit coefficients.
        return np.sum([j * w[j] * np.power(t[i], j - 1) for j in range(1, self.order + 1)])
