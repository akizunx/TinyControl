from .tcconfig import config
from .lti import LinearTimeInvariant, _pickup_dt
from .exception import *
from .lyapunov import *
from .canonical import *
from .fsf import *
import numpy as np
from numpy.linalg import inv, matrix_rank, \
    eigvals, LinAlgError
from scipy.linalg import eigvals

__all__ = ["StateSpace", "ss"]


def _check_ss_matrix(A, B, C, D):
    n0 = A.shape[0]
    n1 = A.shape[1]
    n2 = B.shape[0]
    p = B.shape[1]
    n3 = C.shape[1]
    q = C.shape[0]

    if n0 != n1:
        raise ValueError(f'shape of A should be n x n, got {A.shape}')

    if n2 == n3 == n0:
        pass
    else:
        if n2 != n0 and n3 == n0:
            msg = f'B should have same height with A'
        elif n2 == n0 and n3 != n0:
            msg = f'C should have same width with A'
        else:
            msg = f'B should have same height with A, C should have same width with A'
        raise ValueError(msg)

    if D.shape != (q, p):
        raise ValueError(f'shape of D should be ({q}, {p}), got {D.shape}')


def _siso_zero(A, b, c, d):
    n = A.shape[0]
    M = np.concatenate((np.concatenate((A, -c)),
                        np.concatenate((b, -d))), axis=1)
    N = np.zeros_like(M)
    N[0: n, 0: n] = np.eye(n)
    zeros = eigvals(M, N)
    return zeros[zeros != np.inf]


class StateSpace(LinearTimeInvariant):
    """
    a class implement the state space model
    """

    def __init__(self, A, B, C, D, *, dt=None):
        # let A, B, C and D convert to numpy ndarray
        if config['use_numpy_matrix']:
            A = np.mat(A)
            B = np.mat(B)
            C = np.mat(C)
            D = np.mat(D)
        else:
            A = np.array(A, ndmin=2)
            B = np.array(B, ndmin=2)
            C = np.array(C, ndmin=2)
            D = np.array(D, ndmin=2)

        # check shapes of matrix A B C D
        _check_ss_matrix(A, B, C, D)

        super().__init__(B.shape[1], C.shape[0], dt)
        self.A = A
        self.B = B
        self.C = C
        self.D = D

    def __str__(self):
        a_str = str(self.A)[1: -1]
        b_str = str(self.B)[1: -1]
        a_row_str = a_str.split('\n ')
        b_row_str = b_str.split('\n ')

        s1 = f'A:{" " * (len(a_row_str[0]))}B:'
        tmp = []
        for x, y in zip(a_row_str, b_row_str):
            tmp.append(f'  {x}  {y}')
        s2 = '\n'.join(tmp)

        dummy_row = np.atleast_2d(self.A[-1])
        c_str = str(np.concatenate((self.C, dummy_row), 0))[1: -1]
        dummy_row = np.atleast_2d(self.B[-1])
        d_str = str(np.concatenate((self.D, dummy_row), 0))[1: -1]
        c_row_str = c_str.split('\n ')
        d_row_str = d_str.split('\n ')

        s3 = f'C:{" " * (len(a_row_str[0]))}D:'
        tmp = []
        for x, y in zip(c_row_str, d_row_str):
            tmp.append(f'  {x}  {y}')
        tmp.pop()
        s4 = '\n'.join(tmp)

        return '\n'.join([s1, s2, s3, s4])

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return np.array_equal(self.A, other.A) and \
               np.array_equal(self.B, other.B) and \
               np.array_equal(self.C, other.C) and \
               np.array_equal(self.D, other.D)

    def __ne__(self, other):
        if self == other:
            return False
        else:
            return True

    def __neg__(self):
        return StateSpace(self.A, self.B, -1 * self.C, -1 * self.D, dt=self.dt)

    def __add__(self, other):
        if not isinstance(other, StateSpace):
            other = _convert_to_ss(other)
        return self.parallel(other)

    def __radd__(self, other):
        other = _convert_to_ss(other)
        return other.parallel(self)

    def __mul__(self, other):
        if not isinstance(other, StateSpace):
            other = _convert_to_ss(other)
        return other.cascade(self)

    def __rmul__(self, other):
        other = _convert_to_ss(other)
        return self.cascade(other)

    def parallel(self, *systems):
        """
        Return the paralleled system according to given systems,
        as the following shows.
        ::

                         _____
                    ----| sys1|----
                    |    -----     |
                    |    _____     |
            u(t) --- ---| sys2|---- --- y(t)
                    |    -----     |
                    |      :       |
                    |      :       |
                    |    _____     |
                    ----| sysn|----
                         -----

        :param systems: systems to be paralleled
        :return: the parallel system
        """
        return super().parallel(*systems)

    def _parallel(self, other):
        n1 = self.A.shape[0]
        n = n1 + other.A.shape[0]
        A = np.zeros((n, n))
        A[0: n1, 0: n1] = self.A
        A[n1:, n1:] = other.A
        B = np.concatenate((self.B, other.B), axis=0)
        C = np.concatenate((self.C, other.C), axis=1)
        D = self.D + other.D

        dt = _pickup_dt(self, other)

        return StateSpace(A, B, C, D, dt=dt)

    def cascade(self, *systems):
        """
        Cascade given system from self to the end of systems,
        as the following shows.
        ::


                    _____    _____           _____
            u(t)---| sys1|--| sys2|-- ... --| sysn|---y(t)
                    -----    -----           -----

        :param systems: systems to be cascaded
        :return: the serial system
        """
        return super().cascade(*systems)

    def _cascade(self, other):
        n1 = self.A.shape[0]
        n = n1 + other.A.shape[0]
        A = np.zeros((n, n))
        A[0: n1, 0: n1] = self.A
        A[n1:, n1:] = other.A
        A[n1:, 0: n1] = other.B @ self.C
        B = np.concatenate((self.B, other.B @ self.D), axis=0)
        C = np.concatenate((other.D @ self.C, other.C), axis=1)
        D = other.D @ self.D

        dt = _pickup_dt(self, other)

        return StateSpace(A, B, C, D, dt=dt)

    def feedback(self, k, sign=-1):
        """

        :param k: state space expression of feedback channel
        :type k: StateSpace
        :param sign: determine positive(1) or negative(-1) feedback
        :type sign: int
        :return: the feedback system
        :rtype: StateSpace
        """
        F = np.eye(self.inputs) - sign * k.D @ self.D
        F_inv = inv(F)
        F_inv_D2 = F_inv @ k.D
        F_inv_C2 = F_inv @ k.C
        F_inv_D2_C1 = F_inv_D2 @ self.C
        F_inv_D2_D1 = F_inv_D2 @ self.D
        signed_B1 = sign * self.B
        signed_D1 = sign * self.D

        A1 = self.A + signed_B1 @ F_inv_D2_C1
        A2 = signed_B1 @ F_inv_C2
        A3 = k.B @ (self.C + signed_D1 @ F_inv_D2_C1)
        A4 = k.A + sign * k.B @ self.D @ F_inv_C2
        A = np.concatenate((np.concatenate((A1, A3)),
                            np.concatenate((A2, A4))), axis=1)

        B1 = self.B + signed_B1 @ F_inv_D2_D1
        B2 = k.B @ self.D + sign * k.B @ self.D @ F_inv_D2_D1
        B = np.concatenate((B1, B2))

        C1 = self.C + signed_D1 @ F_inv_D2_C1
        C2 = signed_D1 @ F_inv_C2
        C = np.concatenate((C1, C2), axis=1)

        D = self.D + signed_D1 @ F_inv_D2_D1

        return StateSpace(A, B, C, D)

    def evalfr(self, frequency):
        """
        C * (frequency * I - A )^(-1) * B + D

        :param frequency: the frequency
        :type frequency: complex
        :return:
        :rtype:
        """
        try:
            return (self.C @
                    inv(frequency * np.eye(self.A.shape[0]) - self.A) @
                    self.B + self.D)[0, 0]
        except LinAlgError:
            return float('inf')

    def pole(self):
        """
        Return the poles of the system.

        :return: poles of the system
        :rtype: np.array
        """
        return eigvals(self.A)

    def zero(self, iu=None):
        """
        Return the zeros of the system.

        :return: zeros of the system
        :rtype: np.ndarray
        """
        if self.is_siso:
            return _siso_zero(self.A, self.B, self.C, self.D)
        else:
            raise NotImplementedError

    def ctrb_mat(self):
        """
        Calculate and return the matrix [B A*B A^2*B ... A^(n-1)*B].

        :return: the previous matrix
        :rtype: np.matrix | np.ndarray
        """
        cmat = ctrb_mat(self.A, self.B)

        if config['use_numpy_matrix']:
            return np.mat(cmat)
        else:
            return cmat

    def ctrb_trans_mat(self):
        return ctrb_trans_mat(self.A, self.B)

    def ctrb_form(self):
        T = self.ctrb_trans_mat()
        T_I = inv(T)
        A = T @ self.A @ T_I
        B = T @ self.B
        C = self.C @ T_I

        return StateSpace(A, B, C, self.D, dt=self.dt)

    @property
    def is_controllable(self):
        """
        Return the rank of the controllability matrix.

        :return: if system is controllable return True
        :rtype: bool
        """
        if matrix_rank(self.ctrb_mat()) == self.A.shape[0]:
            return True
        else:
            return False

    def obsv_mat(self):
        """
        Calculate and return the matrix
        ::

            [C        ]
            [C*A      ]
            [C*A^2    ]
            [   ...   ]
            [C*A^(n-1)]

        :return: the previous matrix
        :rtype: np.matrix | np.ndarray
        """
        omat = obsv_mat(self.A, self.C)

        if config['use_numpy_matrix']:
            return np.mat(omat)
        else:
            return omat

    @property
    def is_observable(self):
        """

           the rank of the observability matrix

        :return: if system is observable return True
        :rtype: bool
        """
        if matrix_rank(self.ctrb_mat()) == self.A.shape[0]:
            return True
        else:
            return False

    @property
    def is_gain(self):
        return self.A == self.B == self.C == np.mat(0)

    def place(self, poles):
        """
        Configure system poles by using state feedback.

        The feedback matrix K is calculated by A - B*K.

        :param poles: expected system poles
        :type poles: array_like
        :return: the feedback matrix K
        :rtype: np.matrix | np.ndarray
        """
        K = place(self.A, self.B, poles)
        if config['use_numpy_matrix']:
            return np.mat(K)
        else:
            return K

    @classmethod
    def dual_system(cls, sys_):
        """
        Generate the dual system.

        :param sys_: state space to be calculated
        :type sys_: StateSpace

        :return: the dual system
        :rtype: StateSpace
        """
        return cls(sys_.A.T.copy(), sys_.C.T.copy(), sys_.B.T.copy(), sys_.D.T.copy(),
                   dt=sys_.dt)

    def lyapunov(self, Q=None):
        """
        Solve the equation::
            continuous system: A.T * X + X * A = -I
            discrete system: A.T * X * A - X = -I

        Use sympy to generate a matrix like following one
        ::

            P = [p_00 p_01 ... p_0n]
                [p_01 p_11 ... p_1n]
                [p_02 p_12 ... p_2n]
                [.... .... ... ....]
                [p_0n p_1n ... p_nn]

        P is a symmetric matrix.

        :return: the matrix X or None if there doesn't exist a solve
        :rtype: np.matrix | np.ndarray | None
        """
        if self.is_ctime:
            return lyapunov(self.A, Q)
        else:
            return discrete_lyapunov(self.A, Q)


def ss(*args, **kwargs):
    """
    Create a state space model of the system.

    :param args: A, B, C, D of a system.
                 Or a StateSpace instance.
    :type args:
    :param kwargs: contain sampling time or not
    :type kwargs: dict
    :return: the state space of the system
    :rtype: StateSpace
    """
    from .model_conversion import tf2ss
    length = len(args)
    if length == 1:
        sys_ = args[0]
        try:
            A, B, C, D = sys_.A.copy(), sys_.B.copy(), sys_.C.copy(), sys_.D.copy()
            dt = sys_.dt
        except AttributeError:
            return tf2ss(sys_)
    elif length == 4:
        A, B, C, D = args
        dt = kwargs.get('dt')
    else:
        raise WrongNumberOfArguments("1 or 4 args expected got {0}".format(length))

    return StateSpace(A, B, C, D, dt=dt)


def _convert_to_ss(obj, **kwargs):
    from .model_conversion import tf2ss
    if isinstance(obj, (float, int)):
        inputs = kwargs.get("inputs", 1)
        outputs = kwargs.get("outputs", 1)
        return StateSpace(np.zeros((1, 1)), np.zeros((1, inputs)), np.zeros((outputs, 1)),
                          np.ones((outputs, inputs)) * obj)
    elif issubclass(obj.__class__, LinearTimeInvariant):
        return tf2ss(obj)
    else:
        raise TypeError("wrong type. got {0}".format(type(obj)))
