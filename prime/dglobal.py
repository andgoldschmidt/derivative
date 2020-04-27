from .differentiation import Derivative
from .utils import deriv, integ

import numpy as np
from numpy.linalg import inv
from scipy import interpolate
from scipy.linalg import null_space
from sklearn.linear_model import Lasso


class Spectral(Derivative):
    def __init__(self, **kwargs):
        """
        Compute the numerical derivative by first computing the FFT. In Fourier
        space, derivatives are multiplication by i*phase; compute the IFFT after.

        Args:
            **kwargs: Arbitrary keyword arguments.

        Keyword Args:
            filter: Optional. Maps frequencies to weights in Fourier space.
        """
        # Filter function. Default: Identity filter
        self.filter = kwargs.get('filter', np.vectorize(lambda f: 1))

        self._loaded = False
        self._x_hat = None
        self._freq = None

    def _load(self, t, x):
        self._loaded = True
        self._x_hat = np.fft.fft(x)
        self._freq = np.fft.fftfreq(x.size, d=(t[1] - t[0]))

    def _unload(self):
        self._loaded = False
        self._x_hat = None
        self._freq = None

    def compute(self, t, x, i):
        return next(self.compute_for(t, x, [i]))

    def compute_for(self, t, x, indices):
        self._load(t, x)
        res = np.fft.ifft(1j * 2 * np.pi * self._freq * self.filter(self._freq) * self._x_hat).real
        for i in indices:
            yield res[i]


class Spline(Derivative):
    def __init__(self, s, **kwargs):
        """ Compute the numerical derivative of data x using a (Cubic) spline (the Cubic spline minimizes the curvature of
         the fit). Compute the derivative from the form of the known Spline polynomials.

        Keyword Args:
            order (int): Default is cubic spline (3)
            params['smoothing']: Amount of smoothing
            params['periodic']: Default is False
        """
        self.smoothing = s
        self.order = kwargs.get('order', 3)
        self.periodic = kwargs.get('periodic', False)

        self._loaded = False
        self._t = None
        self._x = None
        self._spl = None

    def load(self, t, x):
        self._loaded = True
        self._t = t
        self._x = x
        # returns (knots, coefficients, order)
        self._spl = interpolate.splrep(self._t, self._x, k=self.order, s=self.smoothing, per=self.periodic)

    def unload(self):
        self._loaded = False
        self._t = None
        self._x = None
        self._spl = None

    def compute(self, t, x, i):
        self.load(t, x)
        return interpolate.splev(self._t[i], self._spl, der=1)

    def compute_for(self, t, x, indices):
        self.load(t, x)
        for i in indices:
            yield interpolate.splev(self._t[i], self._spl, der=1)

    def compute_global(self, t, x):
        self.load(t, x)
        return lambda t0: interpolate.splev(t0, self._spl, der=1)


class TotalVariation(Derivative):
    def __init__(self, order, alpha, **kwargs):
        """ Compute the numerical derivative using Total Squared Variations,

            min_u (1/2)||A u - t||_2^2 + \alpha ||D^order u||_1

        where A is the linear integral operator, D is the linear derivative operator, and T is the time-span of the
        integral.

        If order=1, this is the total-variational derivative. For general order, this is known as the
        polynomial-trend-filtered derivative.

        Args:
            order (int): order of the inner LASSO derivative
            alpha (float): regularization hyper-parameter for LASSO
            **kwargs: keyword arguments for the LASSO function
        """
        self.order = order
        self.alpha = alpha
        self.kwargs = kwargs

        self._loaded = False
        self._t = None
        self._x = None
        self._model = None
        self._res = None

    def load(self, t, x):
        self._loaded = True
        self._t = t
        self._x = x

        # I: Integrals and derivatives
        n = len(self._t)
        dt = self._t[1] - self._t[0]
        # Note: Penalize jump count with + np.identity(n)[:-order, :]
        arrD = deriv(n, dt, order=self.order)
        arrI = integ(n, dt)
        arrI_T = arrI[::-1, ::-1]  # L2 adjoint of integral (slightly different than transpose)

        # II: Compute null space of derivative (analytic solutions are polynomials)
        if self.order == 1:
            nullsp = np.ones([1, n]) / np.sqrt(n)
        elif self.order == 2:
            v1 = np.ones(n) / np.sqrt(n)
            v2 = np.linspace(1, -1, arrD.shape[1])
            nullsp = np.vstack([v1, v2 / np.sqrt(v2 ** 2)])
        elif self.order == 3:
            v1 = np.ones(n) / np.sqrt(n)
            v2 = np.linspace(1, -1, arrD.shape[1])
            v3 = -v2 ** 2 + np.sum(v2 ** 2) / n
            nullsp = np.vstack([v1, v2 / np.sqrt(v2 ** 2), v3 / np.sqrt(v3 ** 2)])
        else:
            null = null_space(arrD).T  # Potential bottleneck! (Requires SVDs)
        Dtilde = np.vstack([arrD, nullsp])

        # III: Compute workers X1 and X2
        invDtilde = inv(Dtilde)
        XDtilde = arrI.dot(invDtilde)
        XDtilde_T = invDtilde.T.dot(arrI_T)
        X1, X2 = XDtilde[:, :-1], XDtilde[:, -1:]
        X1_T, X2_T = XDtilde_T[:-1, :], XDtilde_T[-1:, :]

        # IV: Compute projectors
        # Note: X2.T@X2 is always small
        X2_proj = inv(X2_T.dot(X2)).dot(X2_T)
        P = X2.dot(X2_proj)
        proj = np.identity(n) - P  # Note: proj.T = proj

        # V: LASSO parameters; fit
        A = proj.dot(X1)
        # A_T = X1_T.dot(proj) ...can't give this to LASSO, unfortunately
        b = proj.dot(self._x - self._x[0])
        self._model = Lasso(alpha=self.alpha / n, fit_intercept=False, **self.kwargs)
        self._model.fit(A, b)

        # VI: Restore desired variables
        theta1_hat = self._model.coef_
        theta2_hat = X2_proj.dot(self._x - self._x[0] - X1.dot(theta1_hat))
        self._res = invDtilde.dot(np.hstack([theta1_hat, theta2_hat]))

    def unload(self):
        self._loaded = False
        self._t = None
        self._x = None
        self._model = None

    def compute(self, t, x, i):
        self.load(t, x)
        return self._res[i]

    def compute_for(self, t, x, indices):
        self.load(t, x)
        for i in indices:
            yield self._res[i]
