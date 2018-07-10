from unittest import TestCase
import numpy as np
from tcontrol.statespace import *
from tcontrol.transferfunction import tf, ss2tf
from ..exception import WrongNumberOfArguments


class TestStateSpace(TestCase):
    def setUp(self):
        self.A = np.matrix([[0, 1], [-4, -0.5]])
        self.B = np.matrix([[0.], [1.]])
        self.C = np.matrix([[4., 0.]])
        self.D = np.matrix([0.])
        self.tf_ = tf([4], [1, 0.5, 4])
        self.ss_ = StateSpace(self.A, self.B, self.C, self.D)

    def test___init__(self):
        ss_ = StateSpace(self.A, self.B, self.C, self.D)
        self.assertTrue(ss_.A is self.A)
        self.assertEqual(StateSpace(self.A, self.B, self.C, 0),
                         StateSpace(self.A, self.B, self.C, self.D))
        self.assertRaises(ValueError, StateSpace, self.A, self.C, self.B, 0)
        self.assertRaises(ValueError, StateSpace, self.A, self.C, 0, self.B)

    def test___str__(self):
        pass

    def test___add__(self):
        ss_1 = self.ss_ + self.ss_
        A = [[0, 1, 0, 0], [-4, -.5, 0, 0], [0, 0, 0, 1], [0, 0, -4, -.5]]
        B = [[0], [1], [0], [1]]
        C = [4, 0, 4, 0]
        ss_2 = StateSpace(A, B, C, 0)
        self.assertEqual(ss_1, ss_2)

    def test___mul__(self):
        pass

    def test_feedback(self):
        A = [[0.814723686393179, 0.913375856139019, 0.278498218867048],
             [0.905791937075619, 0.632359246225410, 0.546881519204984],
             [0.126986816293506, 0.0975404049994095, 0.957506835434298]]
        B = [[0.964888535199277, 0.957166948242946],
             [0.157613081677548, 0.485375648722841],
             [0.970592781760616, 0.800280468888800]]
        C = [[0.141886338627215, 0.915735525189067, 0.959492426392903],
             [0.421761282626275, 0.792207329559554, 0.655740699156587]]
        D = [[0.0357116785741896, 0.933993247757551],
             [0.849129305868777, 0.678735154857774]]
        s1 = ss(A, B, C, D)
        A_ = [[0.757740130578333, 0.655477890177557, 0.0318328463774207],
              [0.743132468124916, 0.171186687811562, 0.276922984960890],
              [0.392227019534168, 0.706046088019609, 0.0461713906311539]]
        B_ = [[0.0971317812358475, 0.317099480060861],
              [0.823457828327293, 0.950222048838355],
              [0.694828622975817, 0.0344460805029088]]
        C_ = [[0.438744359656398, 0.765516788149002, 0.186872604554379],
              [0.381558457093008, 0.795199901137063, 0.489764395788231]]
        D_ = [[0.445586200710900, 0.709364830858073],
              [0.646313010111265, 0.754686681982361]]
        s2 = ss(A_, B_, C_, D_)
        sys_ = s1.feedback(s2)
        ans_b = np.array([[0.497199047022938, 0.241439722567989],
                         [0.00397672393087681, 0.227709022266958],
                         [0.538296661406946, 0.149857420314700],
                         [0.132223192796285, 0.0885007994366726],
                         [0.293713307787898,0.543698319397011],
                         [-0.117388364568095, 0.367561733989439]])
        ans_d = np.array([[-0.192541214208884, 0.523103714748680],
                          [0.475954939644619, 0.118861134193000]])

        self.assertTrue(np.all(np.less_equal(np.abs(sys_.B - ans_b), 1e-6)))
        self.assertTrue(np.all(np.less_equal(np.abs(sys_.D - ans_d), 1e-6)))

    def test_pole(self):
        self.assertTrue(np.array_equal(self.ss_.pole(), ss2tf(self.ss_).pole()))

    def test_controllability(self):
        pass

    def test_is_controllable(self):
        self.assertTrue(self.ss_.is_controllable())

    def test_observability(self):
        pass

    def test_is_observable(self):
        self.assertTrue(self.ss_.is_observable())

    def test_dual_system(self):
        _ = StateSpace.dual_system(self.ss_)
        self.assertTrue(np.all(np.equal(_.A.T, self.ss_.A)))
        self.assertTrue(np.all(np.equal(_.C.T, self.ss_.B)))
        self.assertTrue(np.all(np.equal(_.B.T, self.ss_.C)))

    def test_to_controllable_form(self):
        T = self.ss_.to_controllable_form()
        A = T.I*self.A*T
        B = T.I*self.B
        C = self.C*T
        ss_1 = StateSpace(A, B, C, 0)
        ss_2 = StateSpace([[0, 1], [-4, -.5]], [[0], [1]], [4, 0], 0)
        self.assertEqual(ss_1, ss_2)

    def test_ss(self):
        self.assertEqual(ss(self.A, self.B, self.C, self.D), self.ss_)
        self.assertRaises(WrongNumberOfArguments, ss, self.A, self.B, self.C, self.D,
                          self.B)
        self.assertEqual(ss(self.tf_), self.ss_)

    def test_tf2ss(self):
        ss_ = tf2ss(self.tf_)
        self.assertTrue(np.all(ss_.A == self.A))
        self.assertTrue(np.all(ss_.B == self.B))
        self.assertTrue(np.all(ss_.C == self.C))
        self.assertRaises(TypeError, tf2ss, ss_)

    def test_ss2tf(self):
        self.assertEqual(ss2tf(self.ss_), self.tf_)

    def test_continuous_to_discrete(self):
        A = np.array([[0, 1], [0, -2]])
        B = np.array([[0], [1]])
        sys_ = StateSpace(A, B, self.C, self.D)
        d_sys_ = continuous_to_discrete(sys_, 0.05)
        continuous_to_discrete(d_sys_, 0.01)
        self.assertWarns(UserWarning, continuous_to_discrete, d_sys_, 0.01)

    def test_lyapunov(self):
        A = [[0, 1], [-2, -3]]
        B = [[0], [1]]
        C = [1, 0]
        ss_ = StateSpace(A, B, C, 0)
        self.assertTrue(np.array_equal(lyapunov(ss_), [[1.25, 0.25], [0.25, 0.25]]))
        system = tf2ss(tf([0.5], [1, 1, 0]))
        self.assertEqual(None, lyapunov(system))

    def test_discretize(self):
        d_sys = StateSpace.discretize(self.ss_, 1)
        error = np.abs(d_sys.A - [[-0.2231, 0.3594], [-1.4376, -0.4028]])
        self.assertTrue(np.all(np.less_equal(error, 1e-4)))
        error = np.abs(d_sys.B - [[0.3057], [0.3593]])
        self.assertTrue(np.all(np.less_equal(error, 1e-4)))
