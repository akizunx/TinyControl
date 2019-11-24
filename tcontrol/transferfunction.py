import numbers
from collections import Iterable
from copy import deepcopy
from itertools import combinations

import numpy as np
import sympy as sym

from .exception import *
from .lti import LinearTimeInvariant, _pickup_dt

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
        *shape1, depth1 = _get_tf_structure(num)
        *shape2, depth2 = _get_tf_structure(den)

        numerator = deepcopy(num)
        denominator = deepcopy(den)
        for j in range(shape1[1]):
            for i in range(shape2[0]):
                # discarding leading zeros in numerator and denominator
                if depth1 == 1:
                    n = np.trim_zeros(num, 'f')
                    d = np.trim_zeros(den, 'f')
                    numerator = [[np.array(n)]]
                    denominator = [[np.array(d)]]
                else:
                    n = np.trim_zeros(num[i][j], 'f')
                    d = np.trim_zeros(den[i][j], 'f')
                    numerator[i][j] = np.array(n)
                    denominator[i][j] = np.array(d)

        super().__init__(shape1[1], shape1[0], dt)
        self._num = numerator
        self._den = denominator

    def __getitem__(self, item):
        if isinstance(item, tuple):
            if isinstance(item[0], numbers.Integral):
                i, j = item
                return TransferFunction(self._num[i][j], self._den[i][j], dt=self.dt)
            elif isinstance(item[0], slice):
                i, j = item
                num = [x[j] for x in self._num[i]]
                den = [x[j] for x in self._den[i]]
                return TransferFunction(num, den, dt=self.dt)
        else:
            raise NotImplementedError

    def __str__(self):
        if self.is_siso:
            gs, *_ = _tf_to_symbol(self.num, self.den)
            gs = str(gs).replace('(', '').replace(')', '')

            if self.is_gain:
                return f'static gain: {gs}'
            else:
                cs, rs = gs.split('/')
                len1 = len(cs)
                len2 = len(rs)
                len_diff = len2 - len1
                indent = ' ' * (len_diff // 2) if not len_diff % 2 else ' ' * (
                        len_diff // 2 + 1)
                r = f'{indent}{cs}\n{"-" * len2}\n{rs}'

                if self.is_ctime:
                    return r
                else:
                    return r.replace('s', 'z') + f'\nsample time:{self.dt}s'
        else:
            # TODO: design mimo display style
            return str((str(self._num) + '\n' + str(self._den)))

    __repr__ = __str__

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
        return self.parallel(other)

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

        return other.cascade(self)

    __rmul__ = __mul__

    def __matmul__(self, other):
        return self.cascade(other)

    @property
    def is_gain(self):
        return np.poly1d(self.num).order == np.poly1d(self.den).order == 0

    @property
    def num(self):
        if self.is_siso:
            return self._num[0][0]
        else:
            return self._num

    @property
    def den(self):
        if self.is_siso:
            return self._den[0][0]
        else:
            return self._den

    def evalfr(self, frequency):
        val = np.zeros((self.outputs, self.inputs), dtype=np.complex)
        with np.errstate(divide='ignore'):
            for j in range(self.outputs):
                for i in range(self.inputs):
                    val[j, i] = np.polyval(self._num[j][i], frequency) / \
                                np.polyval(self._den[j][i], frequency)

            if self.is_siso:
                return val[0, 0]
            else:
                return val

    def pole(self):
        """
        Return poles of the system.

        :return: poles of the system
        :rtype: numpy.ndarray
        """
        if self.is_siso:
            return np.roots(self.den)
        else:
            # if inputs and outputs are large, this method takes too long to finish
            TFM = _tfm2sympy(self.num, self.den)
            lcm = _get_tfm_lcm(TFM)
            _, ploys = sym.factor_list(lcm)
            poles = []
            for p, _ in ploys:
                poles.extend(np.roots(p.all_coeffs()))
            return np.array(poles)

    def zero(self):
        """
        Return zeros of the system.

        :return: zeros of the system
        :rtype: numpy.ndarray
        """
        if self.is_siso:
            return np.roots(self.num)
        else:
            raise NotImplementedError

    def parallel(self, *systems):
        return super().parallel(*systems)

    def _parallel(self, other):
        dt = _pickup_dt(self, other)
        num = _dummy_tfm(self.inputs, self.outputs)
        den = _dummy_tfm(self.inputs, self.outputs)
        for i in range(self.inputs):
            for j in range(self.outputs):
                left_num, left_den = self._num[j][i], self._den[j][i]
                right_num, right_den = other._num[j][i], other._den[j][i]
                n, d = _add_poly_frac(left_num, left_den, right_num, right_den)
                num[j][i], den[j][i] = n, d

        return TransferFunction(num, den, dt=dt)

    def cascade(self, *systems):
        return super().cascade(*systems)

    def _cascade(self, other):
        dt = _pickup_dt(self, other)
        num = _dummy_tfm(self.inputs, other.outputs)
        den = _dummy_tfm(self.inputs, other.outputs)

        for j in range(self.inputs):
            for i in range(other.outputs):
                n = [x[j] for x in self._num]
                d = [x[j] for x in self._den]
                nn, dd = _poly_inner_prod(zip(other._num[i], other._den[i]), zip(n, d))
                num[i][j] = nn
                den[i][j] = dd

        return TransferFunction(num, den, dt=dt)

    def feedback(self, other=1, sign=-1):
        """
        Add the feedback channel.

        :param other: the transfer function of the feedback path
        :type other: TransferFunction | int
        :param sign: if sign is -1 function will create the negative feedback.
                     if sign is 1, the positive feedback.
        :type sign: int
        :return: original system with a feedback channel
        :rtype: TransferFunction
        """
        if other == 1:
            other = TransferFunction([1], [1], dt=self.dt)

        dt = _pickup_dt(self, other)

        num = np.convolve(self.num, other.den)
        den = np.polyadd(np.convolve(self.num, other.num),
                         np.convolve(self.den, other.den) * (-sign))

        return TransferFunction(num, den, dt=dt)


def _get_tf_structure(lst):
    if len(lst) == 0:
        raise ValueError('Empty list!')

    i = 0
    j = 0
    depth = 3
    for i, x in enumerate(lst):
        if not isinstance(x, Iterable):
            depth = 1
            break
        for j, y in enumerate(x):
            if not isinstance(y, Iterable):
                raise ValueError('Matrix of numbers is not valid')

    return i + 1, j + 1, depth


def _dummy_tfm(inputs, outputs):
    return [[(j, i) for i in range(inputs)] for j in range(outputs)]


def _poly_inner_prod(a, b):
    n = np.array([0])
    d = np.array([1])
    for (n1, d1), (n2, d2) in zip(a, b):
        n, d = _add_poly_frac(n, d, np.convolve(n1, n2), np.convolve(d1, d2))
    return n, d


def _add_poly_frac(left_num, left_den, right_num, right_den):
    if np.array_equal(left_den, right_den):
        d = left_den
        n = np.polyadd(left_num, right_num)
    else:
        d = np.convolve(left_den, right_den)
        n = np.polyadd(np.convolve(left_num, right_den),
                       np.convolve(right_num, left_den))
    return n, d


def _tfm2sympy(num, den):
    tfm = _dummy_tfm(len(num[0]), len(num))
    s = sym.symbols('s')
    for j, (n, d) in enumerate(zip(num, den)):
        for i, (a, b) in enumerate(zip(n, d)):
            tfm[j][i] = sym.Poly.from_list(a, s) / sym.Poly.from_list(b, s)
    return sym.Matrix(tfm)


def _get_nth_minors(M, n):
    q, p = M.shape
    if n > min(q, p):
        raise ValueError(f"Matrix M doesn\'t have {n}th minors")

    minors = []
    for j in combinations(range(q), n):
        for i in combinations(range(p), n):
            minor = sym.det(M[j, i]).together()
            minors.append(minor)
    return minors


def _get_tfm_lcm(TFM):
    lcms = []
    for i in range(1, min(TFM.shape) + 1):
        minors = _get_nth_minors(TFM, i)
        dens = []
        for minor in minors:
            _, den = sym.fraction(minor)
            dens.append(den)
        lcms.append(sym.lcm_list(dens))
    lcm = sym.Poly(sym.lcm_list(lcms))
    return lcm


def _tf_to_symbol(num, den):
    s = sym.Symbol('s')
    cs = sym.Poly.from_list(num, gens=s)
    rs = sym.Poly.from_list(den, gens=s)
    gs = cs / rs
    return gs, cs, rs


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
                 s + 1
        ----------------------
        1.0*s**2 + 0.5*s + 1.0
        >>> tf(system)
                 s + 1
        ----------------------
        1.0*s**2 + 0.5*s + 1.0
        >>> tf([1], [1, 1], 1)
          1
        -----
        z + 1
        sample time:1s
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
            msg = f'type of arg should be TransferFunction, got {type(args[0])}'
            raise TypeError(msg)
    elif length == 0:
        num = kwargs.get('num')
        den = kwargs.get('den')
        dt = kwargs.get('dt')
        if num is None or den is None:
            msg = f'cannot find num, den, [dt] in kwargs, ' \
                  f'got num: {num}, den: {den}, dt: {dt}'
            raise WrongNumberOfArguments(msg)
    else:
        raise WrongNumberOfArguments(f'1, 2 or 3 arg(s) expected. received {length}')

    return TransferFunction(num, den, dt=dt)


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
    num = np.poly(z)
    num = np.polymul([k], num)
    den = np.poly(p)
    return TransferFunction(num, den)
