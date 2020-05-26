API Reference 
=============

There are two ways to interact with the code. The first way is to do a specific import of the desired Derivative object and explicitely construct the implementation,

.. code-block:: python

	from primelab import FiniteDifference
	result = FiniteDifference(k=1).d(x,t)

The second way is top use the functional interface and pass the kind of derivative as an argument along with any required parameters,

.. code-block:: python

	from primelab import dxdt
	result = dxdt(x, t, "finite_difference", k=1)

Main module (primelab.differentiation)
--------------------------------------

dxdt (functional interface)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: primelab.differentiation.dxdt

Derivative (base class)
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: primelab.differentiation.Derivative
    :members:
