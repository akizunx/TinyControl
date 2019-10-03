Library Reference
=================

.. automodule:: tcontrol

LTI
-----------------------

.. currentmodule:: tcontrol.lti

Class
>>>>>

.. autoclass:: LinearTimeInvariant
   :members:
   :private-members:

Function
>>>>>>>>

.. autofunction:: isctime
.. autofunction:: isdtime
.. autofunction:: issiso

Transfer Function Model
-----------------------

.. currentmodule:: tcontrol.transferfunction

Class
>>>>>

.. autoclass:: TransferFunction
   :members:
   :private-members:
   :special-members:

Function
>>>>>>>>

.. autofunction:: tf
.. autofunction:: zpk

State Space Model
-----------------

.. currentmodule:: tcontrol.statespace

Class
>>>>>

.. autoclass:: StateSpace
   :members:
   :private-members:
   :special-members:

Function
>>>>>>>>

.. autofunction:: ss
.. autofunction:: lyapunov
.. autofunction:: place

Model Conversion
----------------

.. py:currentmodule:: tcontrol.model_conversion
.. autofunction:: tf2ss
.. autofunction:: ss2tf

Time Response
-------------

.. currentmodule:: tcontrol.time_response
.. autofunction:: step
.. autofunction:: impulse
.. autofunction:: ramp
.. autofunction:: any_input

Frequency Response
------------------

.. py:currentmodule:: tcontrol.frequency
.. autofunction:: nyquist
.. autofunction:: bode

Discretization
--------------

.. py:currentmodule:: tcontrol.discretization
.. autofunction:: c2d


Lyapunov Equation
-----------------

.. py:currentmodule:: tcontrol.lyapunov
.. autofunction:: lyapunov
.. autofunction:: discrete_lyapunov
