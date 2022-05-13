API Reference 
=============

There are two ways to interact with the code. The first way is to do a specific import of the desired derivative object
and explicitely construct the implementation,

.. code-block:: python

	from derivative import FiniteDifference
	result = FiniteDifference(k=1).d(x,t)

The second way is top use the functional interface and pass the kind of derivative as an argument along with any
required parameters,

.. code-block:: python

	from derivative import dxdt
	result = dxdt(x, t, "finite_difference", k=1)

|
|

Main module (derivative.differentiation)
----------------------------------------

dxdt (functional interface)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: derivative.differentiation.dxdt

Derivative (base class)
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: derivative.differentiation.Derivative
    :members:
