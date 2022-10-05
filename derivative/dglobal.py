from .differentiation import Derivative, register
from .utils import deriv, integ, _memoize_arrays

import numpy as np
from numpy.linalg import inv
from scipy import interpolate, sparse
from scipy.special import legendre
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
        """
        Compute the numerical derivative of data x using a spline. The Cubic spline is the default choice because it
        minimizes the curvature of the fit. Compute the derivative from the form of the fit Spline polynomials.

        Args:
            s (float): Amount of smoothing

        Keyword Args:
            order (int): Default is cubic spline (3). Supports 1 <= order <= 5.
            periodic (bool): Default False.

        Raises:
            TypeError: length of input > self.order must hold for spline interpolation.
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

        .. math ::

            \\min_u \\frac{1}{2} \\|A u - (x-x_0)\\|_2^2 + \\alpha \\|D^{\\textrm{order}+1} u\\|_1

        where A is the linear integral operator, and D is the linear derivative operator. The vector u finds a
        global derivative that is a piecewise function made up of polynomials of the provided order.

        If order=0, this is equivalent to the total-variation derivative (c.f. R. Chartrand, [2]).

        Args:
            order (int): Order of the inner LASSO derivative
            alpha (float): Regularization hyper-parameter for the LASSO problem.
            **kwargs: Keyword arguments for the sklearn LASSO function.

        Raises:
            ValueError: Requires that the number of points n > order + 1 to compute the objective.
            ConvergenceWarning: The Lasso optimization may fail to converge. Try increasing 'max_iter'.
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
        #  Analytic solutions to the null space are Legendre polynomials.
        n = diffop.shape[1]
        x = np.linspace(-1,1,n)
        dx = x[1]-x[0]
        nullsp = []
        for i in range(order):
            lp = legendre(i,monic=True)(x)
            norm = np.sqrt(2/(2*i+1)/dx)
            lp = lp/norm
            nullsp.append(lp)
        nullsp = np.vstack(nullsp)
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


@register("kalman")
class Kalman(Derivative):
    def __init__(self, alpha = 1):
        """ Fit the derivative assuming that given data are noisy measurements

        The Kalman smoother is the maximum likelihood estimator (MLE) for a process whose derivative obeys a Brownian
        motion. Equivalently, it is the MLE for a process whose measurement errors are drawn from standard normal distributions. 
        Specifically, it minimizes the convex objective

        .. math ::
            \\min_{x, \\dot x} \\|Hx-z\\|_{R^{-1}}^2 + \\alpha \\|G(x, \\dot x)\\|_{Q^{-1}}^2

        In this implementation, the we have fixed H = R :math:`\\equiv \\mathbb{1}` and let

        .. math ::
            Q \\equiv \\left[\\begin{array}{cc}\\Delta t & \\Delta t^2/2 \\\\ \\Delta t^2/2 & \\Delta t^3/3\\end{array}\\right].

        Args:
            alpha (float): Ratio of measurement error variance to assumed process variance.
        """
        self.alpha = alpha


    @_memoize_arrays(1)
    def _global(self, t, z, alpha):
        delta_times = t[1:]-t[:-1]
        n = len(t)
        Qs = [np.array([[dt, dt**2/2], [dt**2/2, dt**3/3]]) for dt in delta_times]
        Qinv = alpha*sparse.block_diag([np.linalg.inv(Q) for Q in Qs])
        Qinv = (Qinv + Qinv.T)/2  # force to be symmetric

        G_left = sparse.block_diag([-np.array([[1, 0], [dt, 1]]) for dt in delta_times])
        G_right = sparse.eye(2*(n-1))
        align_cols = sparse.csc_matrix((2 * (n-1), 2))
        G = sparse.hstack((G_left, align_cols)) + sparse.hstack((align_cols, G_right))

        H = sparse.lil_matrix((n, 2*n))
        H[:, 1::2] = sparse.eye(n)

        rhs = H.T @ z.reshape((-1,1))
        lhs = H.T @ H + G.T @ Qinv @ G
        sol = np.linalg.solve(lhs.toarray(), rhs)
        x_hat = (H @ sol).flatten()
        x_dot_hat = (H[:, list(range(1,2*n))+ [0]] @ sol).flatten()
        return x_hat, x_dot_hat

    def compute(self, t, x, i):
        x_dot_hat = self._global(t, x, self.alpha)[1]
        return x_dot_hat[i]

    def compute_for(self, t, x, indices):
        x_dot_hat = self._global(t, x, self.alpha)[1]
        for i in indices:
            yield x_dot_hat[i]

    def compute_x(self, t, x, i):
        x_hat = self._global(t, x, self.alpha)[0]
        return x_hat[i]

    def compute_x_for(self, t, x, indices):
        x_hat = self._global(t, x, self.alpha)[0]
        for i in indices:
            yield x_hat[i]
