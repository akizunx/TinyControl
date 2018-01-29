Advanced Usage
==============

Pole-Zero Map
-------------

    >>> system = tc.zpk([3], [-1, -2+1j, -2-1j], 1)
    >>> p, z = tc.pzmap(system)

.. image:: /image/pzmap.png
    :scale: 60%

Root Locus
----------

    >>> system = tc.tf([1, 2], [1, 2, 1, -1])
    >>> r, k = rlocus(system)

Also it works with state space.
::

    >>> system = tc.tf2ss([1, 2], [1, 2, 1, -1])
    >>> r, k = rlocus(system)

.. image:: /image/rlocus.png
    :scale: 60%
