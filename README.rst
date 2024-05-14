|RTD| |PyPI| |Zenodo| |GithubCI| |LIC|

Numerical differentiation of noisy time series data in python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**derivative** is a Python package for differentiating noisy data. The package showcases a variety of improvements that can be made over finite differences when data is not clean.

Want to see an example of how **derivative** can help? This package is part of **PySINDy** (`github.com/dynamicslab/pysindy <https://github.com/dynamicslab/pysindy/>`_), a sparse-regression framework for discovering nonlinear dynamical systems from data.

This package binds common differentiation methods to a single easily implemented differentiation interface to encourage user adaptation.
Numerical differentiation methods for noisy time series data in python includes:

1. Symmetric finite difference schemes using arbitrary window size.

2. Savitzky-Galoy derivatives (aka polynomial-filtered derivatives) of any polynomial order with independent left and right window parameters.

3. Spectral derivatives with optional filter.

4. Spline derivatives of any order.

5. Polynomial-trend-filtered derivatives generalizing methods like total variational derivatives.

6. Kalman derivatives find the maximum likelihood estimator for a derivative described by a Brownian motion.

7. Kernel derivatives smooth a random process defined by its kernel (covariance).

.. code-block:: python

    from derivative import dxdt
    import numpy as np

    t = np.linspace(0,2*np.pi,50)
    x = np.sin(x)

    # 1. Finite differences with central differencing using 3 points.
    result1 = dxdt(x, t, kind="finite_difference", k=1)

    # 2. Savitzky-Golay using cubic polynomials to fit in a centered window of length 1
    result2 = dxdt(x, t, kind="savitzky_golay", left=.5, right=.5, order=3)

    # 3. Spectral derivative
    result3 = dxdt(x, t, kind="spectral")

    # 4. Spline derivative with smoothing set to 0.01
    result4 = dxdt(x, t, kind="spline", s=1e-2)

    # 5. Total variational derivative with regularization set to 0.01
    result5 = dxdt(x, t, kind="trend_filtered", order=0, alpha=1e-2)

    # 6. Kalman derivative with smoothing set to 1
    result6 = dxdt(x, t, kind="kalman", alpha=1)
    
    # 7. Kernel derivative with smoothing set to 1
    result7 = dxdt(x, t, kind="kernel", sigma=1, lmbd=.1, kernel="rbf")

Contributors:
-------------
Thanks to the members of the community who have contributed!

+-----------------------------------------------------------------+-----------------------------------------------------------------------------------+
|`Jacob Stevens-Haas <https://github.com/Jacob-Stevens-Haas>`_    | Kalman derivatives `#12 <https://github.com/andgoldschmidt/derivative/pull/12>`_, |
|                                                                 | and more!                                                                         |
+-----------------------------------------------------------------+-----------------------------------------------------------------------------------+


References:
-----------

[1] Numerical differentiation of experimental data: local versus global methods- K. Ahnert and M. Abel

[2] Numerical Differentiation of Noisy, Nonsmooth Data- Rick Chartrand

[3] The Solution Path of the Generalized LASSO- R.J. Tibshirani and J. Taylor

[4] A Kernel Approach for PDE Discovery and Operator Learning - D. Long et al.


Citing derivative:
------------------
The **derivative** package is a contribution to `PySINDy <https://github.com/dynamicslab/pysindy/>`_; this work has been published in the Journal of Open Source Software (JOSS). If you use **derivative** in your work, please cite it using the following reference:

Kaptanoglu et al., (2022). PySINDy: A comprehensive Python package for robust sparse system identification. Journal of Open Source Software, 7(69), 3994, https://doi.org/10.21105/joss.03994

.. code-block:: text

	@article{kaptanoglu2022pysindy,
		doi = {10.21105/joss.03994},
		url = {https://doi.org/10.21105/joss.03994},
		year = {2022},
		publisher = {The Open Journal},
		volume = {7},
		number = {69},
		pages = {3994},
		author = {Alan A. Kaptanoglu and Brian M. de Silva and Urban Fasel and Kadierdan Kaheman and Andy J. Goldschmidt and Jared Callaham and Charles B. Delahunt and Zachary G. Nicolaou and Kathleen Champion and Jean-Christophe Loiseau and J. Nathan Kutz and Steven L. Brunton},
		title = {PySINDy: A comprehensive Python package for robust sparse system identification},
		journal = {Journal of Open Source Software}
		}
    

.. |RTD| image:: https://readthedocs.org/projects/derivative/badge/?version=latest
   :target: https://derivative.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
  
.. |LIC| image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://derivative.readthedocs.io/en/latest/license.html
   :alt: MIT License

.. |PyPI| image:: https://badge.fury.io/py/derivative.svg
    :target: https://pypi.org/project/derivative/

.. |Zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.6617446.svg
   :target: https://doi.org/10.5281/zenodo.6617446

.. |GithubCI| image:: https://github.com/andgoldschmidt/derivative/actions/workflows/push-test.yml/badge.svg
    :target: https://github.com/andgoldschmidt/derivative/actions/workflows/push-test.yml

