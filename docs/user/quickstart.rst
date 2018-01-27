Quick Start
===========

The Very Beginning
------------------
First of all, import the module.
::

    >>> import tcontrol as tc

Set Up a System
---------------

Initialize a transfer function model.
::

    >>> system = tc.tf([1], [1, 1])
    >>> print(system)
    1/(s + 1)

Or, you can initialize a state space model.
::

    >>> system = tc.ss([[0, 1], [-2, -3]], [[0], [1]], [1, 0], [[0]])
    >>> print(system)
    A:
    [[ 0  1]
     [-2 -3]]

    B:
    [[0]
     [1]]

    C:
    [[1 0]]

    D:
    [[0]]

Do Some Operations
------------------

Connect two systems in series.
::

    >>> sys1 = tc.tf([1], [1, 1])
    >>> sys2 = tc.tf([2], [2, 1])
    >>> sys3 = sys1 * sys2
    >>> print(sys3)
    2/(2*s**2 + 3*s + 1)

Connect two systems in parallel.
::

    >>> sys1 = tc.tf([1], [1, 1])
    >>> sys2 = tc.tf([2], [2, 1])
    >>> sys3 = sys1 + sys2
    >>> print(sys3)
    (4*s + 3)/(2*s**2 + 3*s + 1)

Add feedback to a system. The default is negative feedback.
::

    >>> sys1 = tc.tf([1], [1, 1])
    >>> sys2 = tc.tf([2], [2, 1])
    >>> sys1.feedback(sys2)
    (2*s + 1)/(2*s**2 + 3*s + 3)

If it is a positive feedback, pass --1 to the function.
::

    >>> sys1.feedback(sys2, -1)
    (2*s + 1)/(-2*s**2 - 3*s + 1)

Evaluate Time Response
----------------------

========= =================
Function        Use
========= =================
step       step response
impulse    impulse response
ramp       ramp response
any_input  any input response
========= =================

Step Response
>>>>>>>>>>>>>

    >>> system = tc.tf([1], [1, 1])
    >>> y, t = tc.step(system)  # y is the system's output

.. image:: /image/step_res.png
    :scale: 60%

Impulse Response
>>>>>>>>>>>>>>>>

    >>> system = tc.tf([1], [1, 1])
    >>> y, t = tc.impulse(system)

.. image:: /image/impulse_res.png
    :scale: 60%

Ramp Response
>>>>>>>>>>>>>

    >>> system = tc.tf([1], [1, 1])
    >>> y, t = tc.ramp(system)

.. image:: /image/ramp_res.png
    :scale: 60%

Any Input Response
>>>>>>>>>>>>>>>>>>
For example, we want to test a system with a sine signal.
::

    >>> import numpy as np  # import numpy to generate time array
    >>> system = tc.tf([1], [1, 1])
    >>> t = np.linspace(0, 10, 1000)
    >>> u = np.sin(t)  # input signal
    >>> y, t = tc.any_input(system, t, u)

.. image:: /image/any_input_res.png
    :scale: 60%

.. note::
    Those functions are also available for state space model.

Get Frequency Response
----------------------

Bode Plot
>>>>>>>>>

    >>> system = tc.zpk([], [0, -1, -2], 2)  # create a system by zeros and poles
    >>> A, phi, omega = tc.bode(system)

.. image:: /image/bode.png
    :scale: 60%

Nyquist Plot
>>>>>>>>>>>>

    >>> system = tc.tf([0.5], [1, 2, 1, 0.5])
    >>> r, omega = tc.nyquist(system)

.. image:: /image/nyquist.png
    :scale: 60%

Conversion Between Two Models
-----------------------------

Use **tf2ss** to convert transfer function to state space.
::

    >>> system = tc.tf([1], [1, 2, 0])
    >>> tc.tf2ss(system)
    A:
    [[ 0.  1.]
     [ 0. -2.]]
    B:
    [[0.]
     [1.]]
    C:
    [[1. 0.]]
    D:
    [[0.]]

Use **ss2tf** to convert state space to transfer function.
::

    >>> A = [[ 0.,  1.], [ 0., -2.]]
    >>> B = [[0], [1]]
    >>> C = [[1, 0]]
    >>> system = tc.ss(A, B, C, 0)
    >>> tc.ss2tf(system)
    1.0/(1.0*s**2 + 2.0*s)
