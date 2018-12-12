from tcontrol.transferfunction import TransferFunction
from tcontrol.lti import LinearTimeInvariant as LTI
from matplotlib import pyplot as plt
import numpy as np

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
        if sys_.is_dtime:
            x = np.linspace(-1, 1, 100)
            y = np.sqrt(1 - x**2)
            plt.plot(x, y, '--', color='#929591')
            plt.plot(x, -y, '--',color='#929591')

        if zero.shape[0]:
            l1 = plt.scatter(np.real(zero), np.imag(zero), s=30, marker='o',
                             color='#069af3')
            plt.legend([l1], ['zero'])
        if pole.shape[0]:
            l2 = plt.scatter(np.real(pole), np.imag(pole), s=30, marker='x',
                             color='#fdaa48')
            plt.legend([l2], ['pole'])

        plt.grid()
        plt.axvline(x=0, color='black')
        plt.axhline(y=0, color='black')
        plt.xlabel('Real Axis')
        plt.ylabel('Imag Axis')
        plt.title(title)
        plt.show()

    return pole, zero
