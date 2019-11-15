from tcontrol.plot_utility import plot_pzmap
from tcontrol.transferfunction import TransferFunction
from tcontrol.lti import LinearTimeInvariant as LTI

__all__ = ["pzmap"]


def pzmap(sys_, title='pole-zero map', *, plot=True):
    """
    Use:
        Draw the pole-zero map

    Example:
        >>> import tcontrol as tc
        >>> system = tc.tf([1], [1, 1, 0, 3])
        >>> tc.pzmap(system)
        >>> tc.plot.show()

    :param sys_: the transfer function of the system
    :type sys_: SISO
    :param title: the title of the pzmap
    :type title: str
    :param plot: if plot is true it will draw the picture
    :type plot: bool
    :return: the poles and zeros of the system
    :rtype: (numpy.ndarray, numpy.ndarray)
    """
    if isinstance(sys_, LTI):
        if not isinstance(sys_, TransferFunction):
            raise NotImplementedError('pzmap currently only for TransferFunction')
    else:
        raise TypeError('sys_ should be LTI or sub class of LTI')

    zero = sys_.zero()
    pole = sys_.pole()

    if plot:
        plot_pzmap(pole, sys_, title, zero)

    return pole, zero
