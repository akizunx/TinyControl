import warnings

from tcontrol.lti import LinearTimeInvariant
import numpy as np
import sympy as sym

__all__ = ["StateSpace", "ss", "tf2ss", "continuous_to_discrete", "lyapunov"]


class StateSpace(LinearTimeInvariant):
    """
    a class implement the state space model
    """

    def __init__(self, A, B, C, D, *, dt=None):
        # let A, B, C and D convert to numpy matrix
        if not isinstance(A, np.matrix):
            A = np.mat(A)
        if not isinstance(B, np.matrix):
            B = np.mat(B)
        if not isinstance(C, np.matrix):
            C = np.mat(C)
        if not isinstance(D, np.matrix):
            D = np.mat(D)

        # check shapes of matrix A B C D
        if A.shape[0] != A.shape[1]:
            raise ValueError(
                "{0} != {1}, wrong shape of A".format(A.shape[0], A.shape[1]))
        if B.shape[0] != A.shape[0]:
            raise ValueError(
                "{0} != ({1}, {2}), wrong shape of B".format(B.shape, A.shape[0],
                                                             B.shape[1]))
        if C.shape[1] != A.shape[0]:
            raise ValueError(
                "{0} != ({1}, {2}), wrong shape of C".format(C.shape, A.shape[0],
                                                             C.shape[0]))
        if D.shape != (C.shape[0], B.shape[1]):
            raise ValueError(
                "{0} != ({1}, {2}), wrong shape of D".format(D.shape, C.shape[0],
                                                             B.shape[1]))

        super().__init__(B.shape[1], C.shape[0], dt)
        self.A = A
        self.B = B
        self.C = C
        self.D = D

    def __str__(self):
        a_str = 'A:\n' + str(self.A) + '\n\n'
        b_str = 'B:\n' + str(self.B) + '\n\n'
        c_str = 'C:\n' + str(self.C) + '\n\n'
        d_str = 'D:\n' + str(self.D) + '\n\n'
        return a_str + b_str + c_str + d_str

    __repr__ = __str__

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
        return StateSpace(self.A, self.B, -self.C, -self.D, dt=self.dt)

    def __add__(self, other):
        if self.D.shape != other.D.shape:
            raise ValueError(
                "shapes of D are not equal {0}, {1}".format(self.D.shape,
                                                            other.D.shape))

        if self.dt is None and other.dt is not None:
            dt = other.dt
        elif self.dt is not None and other.dt is None or self.dt == other.dt:
            dt = self.dt
        else:
            raise ValueError("different sampling times")

        A = np.zeros((self.A.shape[0] + other.A.shape[0],
                      self.A.shape[1] + other.A.shape[1]))
        A[0: self.A.shape[0], 0: self.A.shape[1]] = self.A
        A[self.A.shape[0]:, self.A.shape[1]:] = other.A
        B = np.concatenate((self.B, other.B), axis=0)
        C = np.concatenate((self.C, other.C), axis=1)
        D = self.D + other.D
        return StateSpace(A, B, C, D, dt=dt)

    __radd__ = __add__

    def __mul__(self, other):
        if not isinstance(other, StateSpace):
            other = _convert_to_ss(other)

        if self.outputs != other.inputs:
            raise ValueError("outputs are not equal to inputs")

        if self.dt is None and other.dt is not None:
            dt = other.dt
        elif self.dt is not None and other.dt is None or self.dt == other.dt:
            dt = self.dt
        else:
            raise ValueError("Sampling time is different. "
                             "one is {0}, the other is {1}".format(self.dt, other.dt))

        A = np.zeros((self.A.shape[0] + other.A.shape[0],
                      self.A.shape[1] + other.A.shape[1]))
        A[0: self.A.shape[0], 0: self.A.shape[1]] = self.A
        A[self.A.shape[0]:, self.A.shape[1]:] = other.A
        A[0: self.A.shape[0], self.A.shape[0]:] = self.B*other.C
        B = np.concatenate((other.B, self.B*other.D), axis=0)
        C = np.concatenate((self.D*other.C, self.C), axis=1)
        D = self.D*other.D
        return StateSpace(A, C.T, B.T, D, dt=dt)

    def feedback(self, k, sign=-1):
        """

        :param k:
        :type k: StateSpace
        :param sign:
        :type sign:
        :return:
        :rtype:
        """
        F = np.eye(self.inputs) - sign*k.D*self.D
        F_inv = F.I
        F_inv_D2 = F_inv*k.D
        F_inv_C2 = F_inv*k.C

        A1 = self.A + sign*self.B*F_inv_D2*self.C
        A2 = sign*self.B*F_inv_C2
        A3 = k.B*(self.C + sign*self.D*F_inv_D2*self.C)
        A4 = k.A + sign*k.B*self.D*F_inv_C2
        A = np.concatenate((np.concatenate((A1, A3)),
                            np.concatenate((A2, A4))), axis=1)

        B1 = self.B + sign*self.B*F_inv_D2*self.D
        B2 = k.B*self.D + sign*k.B*self.D*F_inv_D2*self.D
        B = np.concatenate((B1, B2))

        C1 = self.C + sign*self.D*F_inv_D2*self.C
        C2 = sign*self.D*F_inv_C2
        C = np.concatenate((C1, C2), axis=1)

        D = self.D + sign*self.D*F_inv_D2*self.D

        return StateSpace(A, B, C, D)

    def pole(self):
        """
        Get the poles of the system.

        :return: poles of the system
        :rtype: np.array
        """
        return np.linalg.eigvals(self.A)

    def controllability(self):
        """
        Calculate and return the matrix [B A*B A^2*B ... A^(n-1)*B].

        :return: the previous matrix
        :rtype: np.matrix
        """
        tmp = self.B.copy()
        for i in range(1, self.A.shape[0]):
            tmp = np.concatenate((tmp, self.A**i*self.B), axis=1)
        return tmp

    def to_controllable_form(self):
        M = self.controllability().I
        p = np.asarray(M[-1]).reshape(-1)
        T = [np.asarray(p*self.A**i).reshape(-1) for i in range(self.A.shape[0])]
        T = np.mat(T)
        T = np.linalg.inv(T)
        return T

    def is_controllable(self):
        """
        Return the rank of the controllability matrix.

        :return: if system is controllable return True
        :rtype: bool
        """
        if np.linalg.matrix_rank(self.controllability()) == self.A.shape[0]:
            return True
        else:
            return False

    def observability(self):
        """
        Calculate and return the matrix
        ::

            [C        ]
            [C*A      ]
            [C*A^2    ]
            [   ...   ]
            [C*A^(n-1)]

        :return: the previous matrix
        :rtype: np.matrix
        """
        tmp = self.C.copy()
        for i in range(1, self.A.shape[0]):
            tmp = np.concatenate((tmp, self.C*self.A**i), axis=0)
        return tmp

    def is_observable(self):
        """

           the rank of the observability matrix

        :return: if system is observable return True
        :rtype: bool
        """
        if np.linalg.matrix_rank(self.observability()) == self.A.shape[0]:
            return True
        else:
            return False

    def place(self, poles):
        """
        Configure system poles by using state feedback.

        The feedback matrix K is calculated by A - B*K.

        :param poles: expected system poles
        :type poles: array_like
        :return: the feedback matrix K
        :rtype: np.matrix
        """
        T = self.to_controllable_form()
        A = T.I*self.A*T
        p = np.poly(poles)[1:]
        p = p[::-1]
        a = np.asarray(A[-1]).reshape(-1)
        K = p[::-1] + a
        return K*T.I

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

    @staticmethod
    def lyapunov(sys_):
        """
        Solve the equation A.T * X + X * A = -I

        Use sympy to generate a matrix like following one
        ::

            P = [p_00 p_01 ... p_0n]
                [p_10 p_11 ... p_1n]
                [p_20 p_21 ... p_2n]
                [.... .... ... ....]
                [p_n0 p_n1 ... p_nn]

        In fact, P is a symmetric matrix

        :param sys_: system
        :type sys_: StateSpace

        :return: the matrix X
        :rtype: np.matrix
        """
        n = sys_.A.shape[0]
        eye = sym.eye(n)
        p = [[sym.Symbol('p_{0}{1}'.format(i, j)) for i in range(n)] for j in range(n)]
        P = sym.Matrix(p)

        eq = sys_.A.T*P + P*sys_.A + eye

        p_set = sym.solve(eq)
        P = P.evalf(subs=p_set)  # evaluate the matrix P
        X = np.asarray(P.tolist(), dtype=float)
        return np.mat(X)


def place(A, B, poles):
    """
    Configure system poles by using state feedback.

    The feedback matrix K is calculated by A - B*K.

    :param A: system matrix
    :type A: matrix_like
    :param B: input matrix
    :type B: matrix_like
    :param poles: expected system poles
    :type poles: array_like
    :return: feedback matrix K
    :rtype: np.matrix
    """
    A = np.mat(A)
    B = np.mat(B)
    C = np.zeros((1, A.shape[0]))
    D = np.zeros((1, B.shape[1]))
    system = StateSpace(A, B, C, D)
    return system.place(poles)


def lyapunov(sys_):
    """
    Solve the equation A.T * X + X * A = -I

    Use sympy to generate a matrix like following one
    ::

        P = [p_00 p_01 ... p_0n]
            [p_10 p_11 ... p_1n]
            [p_20 p_21 ... p_2n]
            [.... .... ... ....]
            [p_n0 p_n1 ... p_nn]

    In fact, P is a symmetric matrix

    :param sys_: system
    :type sys_: StateSpace

    :return: the matrix X
    :rtype: np.matrix
    """
    return StateSpace.lyapunov(sys_)


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
    length = len(args)
    if length == 1:
        _sys = args[0]
        try:
            A, B, C, D = _sys.A.copy(), _sys.B.copy(), _sys.C.copy(), _sys.D.copy()
        except AttributeError:
            return tf2ss(_sys)
    elif length == 4:
        A, B, C, D = args
    else:
        raise ValueError("1 or 4 args expected got {0}".format(length))
    dt = kwargs.get('dt')

    return StateSpace(A, B, C, D, dt=dt)


def tf2ss(sys_):
    """
    Convert transfer function model to state space model.

    :param sys_: the system
    :type sys_: TransferFunction

    :return: corresponded transfer function model
    :rtype: StateSpace
    """
    try:
        num = np.poly1d(sys_.num)
        den = np.poly1d(sys_.den)
    except AttributeError as e:
        raise TypeError("TransferFunction expected got {0}".format(type(sys_))) from e
    dt = sys_.dt

    if den[den.order] != 1:
        num /= den[den.order]
        den /= den[den.order]
    q, r = num/den
    num = r.coeffs
    bn = q.coeffs
    den = den.coeffs
    D = np.matrix(bn)

    # generate matrix A
    A = np.zeros((1, den.shape[0] - 2))
    A = np.concatenate((A, np.eye(den.shape[0] - 2)), axis=0)
    den = den[1:]
    den = np.mat(den[::-1])
    A = np.concatenate((A, -den.T), axis=1)

    B = np.zeros(A.shape[0])
    B[A.shape[0] - num.shape[0]:] = num
    B = np.mat(B[::-1]).T
    c = np.zeros(A.shape[0])
    c[-1] = 1
    C = np.mat(c)
    return StateSpace(A.T, C.T, B.T, D, dt=dt)


def continuous_to_discrete(sys_, sample_time):
    """
    Convert continuous system to discrete system.

    :param sys_: continuous system
    :type sys_: StateSpace

    :param sample_time: sample time of the discrete system.\
    Time unit is second.
    :type sample_time: int | float

    :return: discrete system
    :rtype: StateSpace
    """
    if sys_.isctime():
        G = sample_time*sys_.A + np.eye(sys_.A.shape[0])
        H = sample_time*sys_.B
        return StateSpace(G, H, sys_.C.copy(), sys_.D.copy(), dt=sample_time)
    else:
        warnings.warn("the system is already a discrete system, no need to convert",
                      stacklevel=2)
        return sys_


def _convert_to_ss(obj, **kwargs):
    if isinstance(obj, (float, int)):
        inputs = kwargs.get("inputs", 1)
        outputs = kwargs.get("outputs", 1)
        return StateSpace(np.matrix(0), np.zeros((1, inputs)), np.zeros(outputs, 1),
                          np.ones((outputs, inputs))*obj)
    elif isinstance(obj, LinearTimeInvariant):
        return tf2ss(obj)
    else:
        raise TypeError("wrong type. got {0}".format(type(obj)))
