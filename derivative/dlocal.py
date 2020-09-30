from .differentiation import Derivative, register
import numpy as np
from numpy.polynomial.polynomial import polyfit


@register("finite_difference")
class FiniteDifference(Derivative):
    def __init__(self, k, **kwargs):
        """
        Compute the symmetric numerical derivative of equally-spaced data using the Taylor series. Derivatives at the
        boundaries are computed with the available reduction of the window or first order finite difference.

        Finite difference schemes are also available in numpy.gradient and scipy.misc.derivative.

        Args:
            k (int): Interpolate the data with a polynomial through the 2k+1 points x[i-k], ..., x[i], ... x[i+k]
            **kwargs: Optional keyword arguments.

        Keyword Args:
            periodic (bool): If True, wrap the data. Assumes that x[-1 + 1] = x[0]. If False, truncate at edges.
                Default False.
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

            # Check for overflow and exclude.
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
        of the polynomial. The default edge behavior is to truncate the window at the offending edge. The data does
        not need to be sampled at equal timesteps.

        A simple symmetric version of the Savitzky-Galoy filter is available as scipy.signal.savgol_filter.

        Args:
            left (float): Left edge of the window is t-left
            right (float): Right edge of the window is t+right
            order (int):  Order of polynomial. Expects 0 <= order < points in window.
            **kwargs: Optional keyword arguments.

        Keyword Args:
            iwindow (bool): Whether to use an indexed window. If True, left and right act as indicies for t instead
                of as lengths in units of t. Default False.
            periodic (bool): If True, wrap the data. Assumes that x[-1 + 1] = x[0]. If False, truncate at edges.
                Default False.
            period (float): If periodic is true, set this as the period of the data. Default is t[-1]-t[0], which is
                likely an inacurate estimate.

        Raises:
            RankWarning: The fit may be poorly conditioned if order >= points in window.
        """
        self.left = left
        self.right = right
        self.order = order
        self.iwindow = kwargs.get('iwindow', False)
        self.periodic = kwargs.get('periodic', False)
        self.period = kwargs.get('period', None)

    def _nondimensional_window(self, t, i):
        """ Create an indexed window from a dimensional window."""
        if self.periodic:
            tleft = t[i] - self.left
            tright = t[i] + self.right
            period = self.period if self.period else t[-1] - t[0]
            # Find the nearest (signed) point and add attained periods
            ileft = np.argmin((t - tleft) % period)
            ileft = ileft - len(t)*np.abs(t[ileft]-tleft)//period
            iright = np.argmin((t - tright) % period)
            iright = iright + len(t)*np.abs(t[iright] - tright) // period
        else:
            # Find the nearest point
            ileft = np.argmin(np.abs(t - (t[i] - self.left)))
            iright = np.argmin(np.abs(t - (t[i] + self.right)))
        return [int(ileft), int(iright)]

    def compute(self, t, x, i):
        # Default edge behavior is to truncate the window.
        if self.iwindow:
            i_l = int(i - self.left)
            i_r = int(i + self.right)
            if not self.periodic:
                i_l = max(0, i_l)
                i_r = min(i_r, len(t) - 1)
        else:
            i_l, i_r = self._nondimensional_window(t, i)

        # Construct a polynomial in t using least squares regression.
        # Index views must allow for -left to +right, inclusively.
        ii = np.arange(i_l, i_r + 1)
        # Times are not periodic and initial values must be corrected.
        period = t[-1]-t[0] if self.period is None else self.period
        tfit = t[ii % len(t)] + period*(ii//len(t))
        xfit = x[ii % len(t)]
        # Can raise RankWarning if order exceeds points in the window.
        w = polyfit(tfit, xfit, self.order)

        # Compute the derivative from the polyfit coefficients.
        return np.sum([j * w[j] * np.power(t[i], j - 1) for j in range(1, self.order + 1)])
