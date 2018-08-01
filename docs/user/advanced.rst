Advanced Usage
==============

System Stability Analysis
-------------------------
The tcontrol package provides some function for this usage.
::

Pole-Zero Map
>>>>>>>>>>>>>

    >>> system = tc.zpk([3], [-1, -2+1j, -2-1j], 1)
    >>> p, z = tc.pzmap(system)

.. image:: /image/pzmap.png
    :scale: 60%

Root Locus
>>>>>>>>>>

    >>> system = tc.tf([1, 2], [1, 2, 1, -1])
    >>> r, k = tc.rlocus(system)

Also it works with state space.
::

    >>> system = tc.tf2ss(tc.tf([1, 2], [1, 2, 1, -1]))
    >>> r, k = tc.rlocus(system)

.. image:: /image/rlocus.png
    :scale: 60%

..  note::
    Up till now, these two function illustrated above
    are only available for the SISO(single input and single output) system.

Lyapunov Method
>>>>>>>>>>>>>>>
For the state space model, there is a special method to analyze the system.
::

    >>> system = tc.tf2ss(tc.tf([1], [1, 1]))
    >>> tc.lyapunov(system)
    [[0.5]]
