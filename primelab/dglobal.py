from .differentiation import Derivative, register
from .utils import deriv, integ

import numpy as np
from numpy.linalg import inv
from scipy import interpolate
from scipy.linalg import null_space
from sklearn.linear_model import Lasso


@register("spectral")
class Spectral(Derivative):
    def __init__(self, **kwargs):
        """
        Compute the numerical derivative by first computing the FFT. In Fourier space, derivatives are multiplication
        by i*phase; compute the IFFT after.
        Args:
            **kwargs: Optional keyword arguments.

        Keyword Args:
            filter: Optional. A function that takes in frequencies and outputs weights to scale the coefficient at
                the input frequency in Fourier space. Input frequencies are the discrete fourier transform sample
                frequencies associated with the domain variable. Look into python signal processing resources in
                scipy.signal for common filters.

        """
        # Filter function. Default: Identity filter
        self.filter = kwargs.get('filter', np.vectorize(lambda f: 1))
        self._x_hat = None
        self._freq = None

    def _dglobal(self, t, x):
        self._x_hat = np.fft.fft(x)
        self._freq = np.fft.fftfreq(t.size, d=(t[1] - t[0]))

    def compute(self, t, x, i):
        return next(self.compute_for(t, x, [i]))

    def compute_for(self, t, x, indices):
        self._dglobal(t, x)
        res = np.fft.ifft(1j * 2 * np.pi * self._freq * self.filter(self._freq) * self._x_hat).real
        for i in indices:
            yield res[i]


@register("spline")
class Spline(Derivative):
    def __init__(self, s, **kwargs):
        """ Compute the numerical derivative of data x using a (Cubic) spline (the Cubic spline is a natural choice as
         it minimizes the curvature of the fit). Compute the derivative from the form of the known Spline polynomials.

        Args:
            s (float): Amount of smoothing

        Keyword Args:
            order (int): Default is cubic spline (3).
            periodic (bool): Default False.
        """
        self.smoothing = s
        self.order = kwargs.get('order', 3)
        self.periodic = kwargs.get('periodic', False)

        self._t = None
        self._x = None
        self._spl = None

    def _global(self, t, x):
        self._loaded = True
        self._t = t
        self._x = x
        # returns (knots, coefficients, order)
        self._spl = interpolate.splrep(self._t, self._x, k=self.order, s=self.smoothing, per=self.periodic)

    def compute(self, t, x, i):
        self._global(t, x)
        return interpolate.splev(self._t[i], self._spl, der=1)

    def compute_for(self, t, x, indices):
        self._global(t, x)
        for i in indices:
            yield interpolate.splev(self._t[i], self._spl, der=1)


@register("trend_filtered")
class TrendFiltered(Derivative):
    def __init__(self, order, alpha, **kwargs):
        """ Compute the numerical derivative using Total Squared Variations,

            min_u (1/2)||A u - (x-x_0)||_2^2 + \alpha ||D^(order+1) u||_1

        where A is the linear integral operator, and D is the linear derivative operator. The vector u finds a
        global derivative that is a piecewise function made up of polynomials of the provided order.

        If order=0, this is equivalent to the total-variational derivative.

        Args:
            order (int): Order of the inner LASSO derivative
            alpha (float): Regularization hyper-parameter for the LASSO problem.
            **kwargs: Keyword arguments for the sklearn LASSO function.
        """
        self.dorder = order + 1
        self.alpha = alpha
        self.kwargs = kwargs

        self._t = None
        self._x = None
        self._model = None
        self._res = None

    @staticmethod
    def _nullsp(order, diffop):
        #  Analytic solutions to the null space are polynomials.
        n = diffop.shape[1]
        if order == 1:
            nullsp = np.ones([1, n]) / np.sqrt(n)
        elif order == 2:
            v1 = np.ones(n) / np.sqrt(n)
            v2 = np.linspace(1, -1, diffop.shape[1])
            nullsp = np.vstack([v1, v2 / np.sqrt(v2 ** 2)])
        elif order == 3:
            v1 = np.ones(n) / np.sqrt(n)
            v2 = np.linspace(1, -1, diffop.shape[1])
            v3 = -v2 ** 2 + np.sum(v2 ** 2) / n
            nullsp = np.vstack([v1, v2 / np.sqrt(v2 ** 2), v3 / np.sqrt(v3 ** 2)])
        else:
            nullsp = null_space(diffop).T  # Potential bottleneck! (Requires SVDs)
        return nullsp

    def _global(self, t, x):
        self._t = t
        self._x = x

        # I: Integrals and derivatives
        n = len(self._t)
        dt = self._t[1] - self._t[0]
        # Note: Penalize jump count with + np.identity(n)[:-order, :]
        arrD = deriv(n, dt, order=self.dorder)
        arrI = integ(n, dt)

        # II: Append null space of derivative
        Dtilde = np.vstack([arrD, self._nullsp(self.dorder, arrD)])

        # III: Compute workers X1 and X2
        invDtilde = inv(Dtilde)
        XDtilde = arrI.dot(invDtilde)
        X1, X2 = XDtilde[:, :-1], XDtilde[:, -1:]

        # IV: Compute projectors
        # Note: regarding inv, X2.T@X2 is always small
        X2_proj = inv(X2.T.dot(X2)).dot(X2.T)
        proj = np.identity(n) - X2.dot(X2_proj)

        # V: LASSO parameters; fit
        A = proj.dot(X1)
        b = proj.dot(self._x - self._x[0])
        self._model = Lasso(alpha=self.alpha / n, fit_intercept=False, **self.kwargs)
        self._model.fit(A, b)

        # VI: Restore desired variables
        theta1_hat = self._model.coef_
        theta2_hat = X2_proj.dot(self._x - self._x[0] - X1.dot(theta1_hat))
        self._res = invDtilde.dot(np.hstack([theta1_hat, theta2_hat]))

    def compute(self, t, x, i):
        self._global(t, x)
        return self._res[i]

    def compute_for(self, t, x, indices):
        self._global(t, x)
        for i in indices:
            yield self._res[i]
