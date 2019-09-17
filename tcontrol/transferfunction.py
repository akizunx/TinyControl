import numbers
from collections import Iterable

from .lti import LinearTimeInvariant
from .exception import *
import numpy as np
import sympy as sym

# TODO: trans function of MIMO

__all__ = ["TransferFunction", "tf", "zpk"]


class TransferFunction(LinearTimeInvariant):
    """
    a class implement the transfer function model

    :param num: the numerator of transfer function
    :type num: np.ndarray | list
    :param den: the denominator of transfer function
    :type den: np.ndarray | list
    :param dt: sampling time
    :type dt: int | float
    """

    def __init__(self, num, den, *, dt=None):
        num = np.array(np.poly1d(num))
        den = np.array(np.poly1d(den))
        num, den = _poly_simplify(num, den)

        super().__init__(1, 1, dt)
        self.num = num
        self.den = den

    def __str__(self):
        gs, *_ = _tf_to_symbol(self.num, self.den)
        gs = str(gs).replace('(', '').replace(')', '')

        if self.is_gain:
            return f'static gain: {gs}'
        else:
            cs, rs = gs.split('/')
            len1 = len(cs)
            len2 = len(rs)
            len_diff = len2 - len1
            indent = ' ' * (len_diff // 2) if not len_diff % 2 else ' ' * (len_diff // 2 + 1)
            r = f'{indent}{cs}\n{"-" * len2}\n{rs}'

            if self.is_ctime:
                return r
            else:
                return r.replace('s', 'z') + f'sample time:{self.dt}s'

    __repr__ = __str__

    def __eq__(self, other):
        if not isinstance(other, TransferFunction):
            return False

        if not self.is_ctime and self.dt != other.dt:
            return False

        return np.all(np.isclose(other.num, self.num)) and \
               np.all(np.isclose(other.den, self.den))

    def __neg__(self):
        num = -1 * self.num
        return TransferFunction(num, self.den, dt=self.dt)

    def __add__(self, other):
        """

        :param other:
        :type other: TransferFunction | number.Real
        :return:
        :rtype: TransferFunction
        """
        if isinstance(other, numbers.Real):
            other = TransferFunction([other], [1])

        dt = _get_dt(self, other)

        if np.array_equal(self.den, other.den):
            return TransferFunction(np.polyadd(self.num, other.num), self.den, dt=dt)

        den = np.convolve(self.den, other.den)
        num = np.polyadd(np.convolve(self.num, other.den),
                         np.convolve(other.num, self.den))

        return TransferFunction(num, den, dt=dt)

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-other)

    __rsub__ = __sub__

    def __mul__(self, other):
        """

        :param other:
        :type other: TransferFunction | numbers.Real
        :return:
        :rtype: TransferFunction
        """
        if isinstance(other, numbers.Real):
            return TransferFunction(self.num * other, self.den, dt=self.dt)

        num = np.convolve(self.num, other.num)
        den = np.convolve(self.den, other.den)

        num, den = _poly_simplify(num, den)

        dt = _get_dt(self, other)

        return TransferFunction(num, den, dt=dt)

    __rmul__ = __mul__

    @property
    def is_gain(self):
        return np.poly1d(self.num).order == np.poly1d(self.den).order == 0

    def pole(self):
        """
        Return poles of the system.

        :return: poles of the system
        :rtype: numpy.ndarray
        """
        return np.roots(self.den)

    def zero(self):
        """
        Return zeros of the system.

        :return: zeros of the system
        :rtype: numpy.ndarray
        """
        return np.roots(self.num)

    def feedback(self, other=1, sign=1):
        """
        Add the feedback channel.

        :param other: the transfer function of the feedback path
        :type other: TransferFunction | int
        :param sign: if sign is 1 function will create the negative feedback.
                     if sign is -1, the positive feedback
        :type sign: int
        :return: original system with a feedback channel
        :rtype: TransferFunction
        """
        if other == 1:
            other = TransferFunction([1], [1], dt=self.dt)

        dt = _get_dt(self, other)

        num = np.convolve(self.num, other.den)
        den = np.polyadd(np.convolve(self.num, other.num),
                         np.convolve(self.den, other.den) * sign)

        return TransferFunction(num, den, dt=dt)


def _get_dt(sys1, sys2):
    """
    Determine the sampling time of the new system.

    :param sys1: the first system
    :type sys1: TransferFunction
    :param sys2: the second system
    :type sys2: TransferFunction
    :return: sampling time
    :rtype: int | float
    """
    if sys1.dt == sys2.dt or (sys1.dt is not None and sys2.dt is None):
        dt = sys1.dt
    elif sys1.dt is None and sys2.dt is not None:
        dt = sys2.dt
    else:
        raise ValueError(
            'Expected the same sampling time. got sys1:{0} sys2:{1}'.format(sys1.dt,
                                                                            sys2.dt))
    return dt


def _tf_to_symbol(num, den):
    s = sym.Symbol('s')
    cs = sym.Poly.from_list(num, gens=s)
    rs = sym.Poly.from_list(den, gens=s)
    gs = cs / rs
    return gs, cs, rs


def _poly_simplify(num, den):
    _, cs, rs = _tf_to_symbol(num, den)
    r = np.array(sym.gcd(cs, rs).as_poly().all_coeffs()).astype(num.dtype)
    if not np.array_equal(np.array([1]), r):
        num = np.polydiv(num, r)[0]
        den = np.polydiv(den, r)[0]

    return num, den


# def _poly_gcd(a, b):
#     s = sym.Symbol('s')
#     r = sym.gcd(a, b)
#     if r.is_Number:
#         return np.array([r], dtype=float)
#     p = sym.polys.polytools.poly(r)
#     n = 0
#     r = []
#     while True:
#         k = sym.polys.polytools.Poly.coeff_monomial(p, s**n)
#         if k.is_integer:
#             k = int(k)
#         elif k.is_real:
#             k = float(k)
#         elif k.is_complex:
#             k = complex(k)
#         else:
#             raise ValueError('unexpected coeff type')
#
#         if k == 0 and n != 0:
#             break
#         else:
#             r.insert(0, k)
#         n += 1
#     r = np.array(r)
#     return r


def tf(*args, **kwargs):
    """
    Create a transfer function model of a system.

    :param args: pass in num
    :type args: TransferFunction | List[numbers.Real] |
                numbers.Real
    :return: the transfer function of the system
    :rtype: TransferFunction

    :Example:
        >>> from tcontrol import tf
        >>> system = tf([1, 1], [1, 0.5, 1])
        >>> print(system)
        (s + 1)/(1.0*s**2 + 0.5*s + 1.0)
        >>> system = tf(system)
        >>> print(system)
        (s + 1)/(1.0*s**2 + 0.5*s + 1.0)
    """
    length = len(args)
    if length == 2:
        num, den = args
        dt = None
    elif length == 3:
        num, den, dt = args
    elif length == 1:
        try:
            num = args[0].num
            den = args[0].den
            dt = args[0].dt
        except AttributeError:
            raise TypeError(
                "type of arg should be TransferFunction, got {0}".format(type(args[0])))
    elif length == 0:
        num = kwargs.get('num')
        den = kwargs.get('den')
        dt = kwargs.get('dt')
        if num is None and den is None and dt is None:
            raise WrongNumberOfArguments(f'1, 2 or 3 arg(s) expected. received {length}')
    else:
        raise WrongNumberOfArguments(f'1, 2 or 3 arg(s) expected. received {length}')

    sys_ = TransferFunction(num, den, dt=dt)
    return sys_


def zpk(z, p, k):
    """
    Create a transfer function by zeros, poles, and the gain k

    :param z: zeros of a system
    :type z: np.ndarray | Iterable
    :param p: poles of a system
    :type p: np.ndarray | Iterable
    :param k: the gain of the system
    :type k: numbers.Real
    :return: the transfer function of the system
    :rtype: TransferFunction

    :Example:
        >>> from tcontrol import zpk
        >>> system = zpk([], [1, 0.5, 1], 5.2)
        >>> print(system)
        5.2/(1.0*s**3 - 2.5*s**2 + 2.0*s - 0.5)
    """
    num = np.array([k])
    for zi in z:
        num = np.convolve(num, np.array([1, -zi]))

    den = np.array([1])
    for pi in p:
        den = np.convolve(den, np.array([1, -pi]))
    return TransferFunction(num, den)
