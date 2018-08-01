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
.. autofunction:: ss2tf

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
.. autofunction:: tf2ss
.. autofunction:: continuous_to_discrete
.. autofunction:: lyapunov
.. autofunction:: place

Time Response
-------------

.. currentmodule:: tcontrol.time_response
.. autofunction:: step
.. autofunction:: impulse
.. autofunction:: ramp
.. autofunction:: any_input
