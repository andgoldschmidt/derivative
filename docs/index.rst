prime
=====
Numerical differentiation in python
-----------------------------------

.. code-block:: python

    from prime import FiniteDifference
    import numpy as np

    x = np.linspace(0,2*np.pi,100)
    y = np.sin(x)
    # Central differencing
    dydx = FiniteDifference(1).d(y, x)


Numerical differentiation methods for python, including:

1. Symmetric finite difference schemes using arbitrary window size.

2. Savitzky-Galoy derivatives of any polynomial order with independent left and right window parameters.

3. Spectral derivatives with optional filter.

4. Spline derivatives of any order.

5. Polynomial-trend-filtered derivatives generalizing methods like total variational derivatives.

These examples are intended to survey some common differentiation methods. The goal of this package is to bind these
common differentiation methods to an easily implemented differentiation interface to encourage user adaptation.


References:

[1] Numerical differentiation of experimental data: local versus global methods- K. Ahnert and M. Abel

[2] Numerical Differentiation of Noisy, Nonsmooth Data- Rick Chartrand

[3] The Solution Path of the Generalized LASSO- R.J. Tibshirani and J. Taylor


.. toctree::
    :maxdepth: 2
    :caption: Contents:

    notebooks/Examples.ipynb
    modules
    license


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
