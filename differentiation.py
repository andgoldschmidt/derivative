# -*- coding: utf-8 -*-
"""
Created on Wed Sept 25 11:45:00 2019
@author: agoldschmidt

TODO:
- optional (periodic) boundary conditions

"""
import numpy as np
import abc

import scipy as sci
from scipy import interpolate, integrate, optimize


class Derivative(abc.ABC):
    '''
    Object for computing numerical derivatives.

    Notes:
    -   Derivative methods should return np.nan if the
        implementation fails to compute a derivative at the
        desired index.
    -   Methods should fail only for invalid data
    '''
    
    @abc.abstractmethod
    def compute(self, t, x, i):
        '''Compute derivative (dx/dt)[i]

        Returns:
            (dx/dt)[i]
        '''
        
    def compute_for(self, t, x, indices):
        '''Compute derivative (dx/dt)[i] for i in indices. Overload this if
        desiring a more efficient computation over a list of indices.

        Returns:
            iterator over (dx/dt)[i] for i in indices'''
        for i in indices:
            yield self.compute(t, x, i)

# ----------------------------------------------------------

class FiniteDifference(Derivative):
    ''' Compute the numerical derivative of equally-spaced data using
    the Taylor series.
    Arguments:
        params['k']: the number of points around an index to use for the derivative

    '''
    def __init__(self, params):
        try: 
            self.k = params['k']
        except:
            raise ValueError("Derivative FiniteDifference missing required parameter.")

    def compute(self, t, x, i):
        # Check boundaries (don't compute if outside)
        if i - self.k < 0 or i + self.k > len(x) - 1:
            return np.nan

        dt = t[1] - t[0]
        res = []
        coefficient = 1 # combinatorial coeff. from Taylor series
        for j in range(1,self.k+1):
            coefficient *= (self.k-j+1)/(self.k+j)
            alpha_j = 2*np.power(-1, j+1)*coefficient
            res.append(alpha_j*(x[i+j]-x[i-j])/(2*j)/dt)
        return np.sum(res)

# ----------------------------------------------------------

class Spectral(Derivative):
    ''' Compute the numerical derivative by first computing the FFT. In Fourier 
    space, derivatives are multiplication by i*phase; compute the IFFT after.
    Arguments:
        params['filter']: optional, maps frequencies to weights in Fourier space
    '''
    def __init__(self, params):
        # No required parameters.
        # Filter function. Default: Identity filter
        self.filter = params['filter'] if 'filter' in params else np.vectorize(lambda f: 1)

    def load(self, t, x):
        self._loaded = True
        self._x_hat = np.fft.fft(x)
        self._freq = np.fft.fftfreq(x.size, d=(t[1]-t[0]))

    def unload(self):
        self._loaded = False
        self._x_hat = None
        self._freq = None

    def compute(self, t, x, i):
        return next(self.compute_for(t, x, [i]))

    def compute_for(self, t, x, indices):
        self.load(t, x)
        res = np.fft.ifft(1j*2*np.pi*self._freq*self.filter(self._freq)*self._x_hat).real
        for i in indices:
            yield res[i]

# ----------------------------------------------------------

class SavitzkyGolay(Derivative):
    '''Compute the numerical derivative by first finding the best 
    (least-squares) polynomial of order m < 2k using the k points in
    the neighborhood [t-left, t+right]. The derivative is computed 
    from the coefficients of the polynomial.
    Arguments:
        params['left']: left edge of the window is t-left
        params['right']: right edge of the window is t+right
        params['order']: order of polynomial (m < points in window)
    '''  
    def __init__(self, params):
        # Note: Left and right have units (they do not count points)
        try:
            self.left = params['left']
            self.right = params['right']
            self.order = params['order']
        except:
            raise ValueError("Derivative SavitzkyGolay missing required parameter.")

    def compute(self, t, x, i):
        '''Requires the (t,x) data to be sorted. '''
        i_l = np.argmin(np.abs(t - (t[i] - self.left)))
        i_r = np.argmin(np.abs(t - (t[i] + self.right)))
        
        # window too sparse. TODO: issue warning.
        if self.order > (i_r - i_l): 
            return np.nan
        
        # Construct polynomial in t and do least squares regression
        try:
            polyn_t = np.array([np.power(t[i_l:i_r], n)
                            for n in range(self.order+1)]).T
            w,_,_,_ = np.linalg.lstsq(polyn_t, x[i_l:i_r], rcond=None)
        except np.linalg.LinAlgError:
            # Failed to converge, return bad derivative
            return np.nan

        # Compute derivative from fit
        return np.sum([j*w[j]*np.power(t[i], j-1)
                       for j in range(1, self.order+1)])

    def compute_for(self, t, x, indices):
        # If the window cannot reach any points, throw an exception
        # (likely the user forgets to rescale the window parameter)
        if min(t[1:] -t[:-1]) > max(self.left, self.right):
            raise ValueError("Found bad window ({}, {}) for x-axis data."
                .format(self.left, self.right))
        for d in super().compute_for(t, x, indices):
            yield d

# ----------------------------------------------------------
# -- Utility matrices --------------------------------------
# -- Used for TotalVariation Derivatives, for example. -----
# ----------------------------------------------------------
def derivative_matrix(n, dx=1):
    ''' Equi-spaced derivative via the central difference. '''
    if n == 2:
        return np.array([[-1,1],
                         [-1,1]])/dx
    elif n > 2:
        M = [[-1, 1] + [0]*(n-2)]
        for row in range(0,n-2):
            M.append([0]*row + [-1/2,0,1/2] + [0]*(n-3-row))
        M.append([0]*(n-2) + [-1, 1])
        return np.array(M)/dx
    else:
        raise ValueError('Bad length of {}'.format(n))

def integral_matrix(n, dx=1, C=0):
    ''' Equi-spaced anti-derivative via the trapezoid rule '''
    if n == 2:
        return np.array([[0,0],
                         [1,1]])*dx/2 + C
    elif n > 2:
        M = [[0]*n]
        for row in range(0, n-1):
            M.append([1] + [2]*(row) + [1] + [0]*(n-row-2))
        return np.array(M)*dx/2 + C
    else:
        raise ValueError('Bad length of {}'.format(n))
# ----------------------------------------------------------

class TotalVariation(Derivative):
    def __init__(self, params):
        try:
            self.alpha = params['alpha']
        except:
            raise ValueError("Derivative TotalVariation missing required parameter.")

        self._loaded = False
        self._t = None
        self._x = None
        self._D = None
        self._A = None
        self._res = None

    def load(self, t, x):
        self._loaded = True
        self._t = t
        self._x = x
        self._D = derivative_matrix(len(t), t[1]-t[0])
        self._A = integral_matrix(len(t), t[1]-t[0])
        self._res = np.linalg.solve(self._A.T@self._A + self.alpha*self._D.T@self._D, self._A.T@self._x)

    def unload(self):
        self._loaded = False
        self._t = None
        self._x = None
        self._D = None
        self._A = None

    def compute(self, t, x, i):
        self.load(t, x)
        return self._res[i]
        
    def compute_for(self, t, x, indices):
        self.load(t, x)
        for i in indices:
            yield self._res[i]

# ----------------------------------------------------------

class Spline(Derivative):
    '''Compute the numerical derivative of y using a (Cubic) spline (the Cubic 
    spline minimizes the curvature of the fit). Compute the derivative from 
    the form of the known Spline polynomials.
    Arguments:
        params['order']: Default is cubic spline (3)
        params['smoothing']: Amount of smoothing
        params['periodic']: Default is False
    '''
    def __init__(self, params):
        self.order = params['order'] if 'order' in params else 3
        self.periodic = params['periodic'] if 'periodic' in params else False
        
        try:
            self.smoothing = params['smoothing']
        except:
            raise ValueError("Derivative Spline missing required parameter.")

        self._loaded = False
        self._t = None
        self._x = None
        self._spl = None
    
    def load(self, t, x):
        self._loaded = True
        self._t = t
        self._x = x
        # returns (knots, coefficients, order)
        self._spl = interpolate.splrep(self._t, self._x, k=self.order, 
            s=self.smoothing, per=self.periodic)

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
        