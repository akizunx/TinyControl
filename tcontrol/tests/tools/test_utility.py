from numpy.testing import *


def assert_tf_equal(sys1, sys2):
    if (sys1.is_dtime or sys2.is_dtime) and sys1.dt != sys2.dt:
        raise AssertionError
    assert_array_almost_equal(sys1.den, sys2.den)
    assert_array_almost_equal(sys1.num, sys2.num)



def assert_ss_equal(sys1, sys2):
    if (sys1.is_dtime or sys2.is_dtime) and sys1.dt != sys2.dt:
        raise AssertionError
    assert_array_almost_equal(sys1.A, sys2.A)
    assert_array_almost_equal(sys1.B, sys2.B)
    assert_array_almost_equal(sys1.C, sys2.C)
    assert_array_almost_equal(sys1.D, sys2.D)
