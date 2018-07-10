from functools import singledispatch
import numbers

from .transferfunction import TransferFunction
from .statespace import StateSpace, continuous_to_discrete

__all__ = ['c2d']


@singledispatch
def c2d(sys_, sample_time, method='zoh'):
    """
    Convert a continuous system to a discrete system.

    :param sys_: the system to be transformed
    :type sys_: TransferFunction | StateSpace
    :param sample_time: sample time
    :type sample_time: numbers.Real
    :param method: method used in transformation
    :type method: str
    :return: a discrete system
    :rtype: TransferFunction | StateSpace

    :raises TypeError: Raised when the type of sys_ is wrong

    :Example:

        >>> system = TransferFunction([1], [1, 1])
        >>> c2d(system, 0.5, 'matched')
             0.393469340287367
        -------------------------
        1.0*z - 0.606530659712633
        sample time:0.5s
    """
    raise TypeError(f'TransferFunction or StateSpace expected, got{type(sys_)}')


@c2d.register(TransferFunction)
def f(sys_, sample_time, method):
    return TransferFunction.discretize(sys_, sample_time, method)


@c2d.register(StateSpace)
def f(sys_, sample_time, method):
    return continuous_to_discrete(sys_, sample_time)
