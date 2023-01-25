from double_pendulum import *
import numpy as np
import pytest

@pytest.mark.parametrize(
    "theta1, theta2, expected",
    [
        (  0,   0,            0),
        (  0, 0.5,  3.386187037), 
        (0.5,   0, -7.678514423),
        (0.5, 0.5, -4.703164534),
    ]
)
def test_domega1_dt(theta1, theta2, expected):
    dp = DoublePendulum()
    t = 0
    y = (theta1, 0.25, theta2, 0.15)
    dtheta1_dt, domega1_dt, _, _ = dp(t, y)
    assert np.isclose(dtheta1_dt, 0.25)
    assert np.isclose(domega1_dt, expected)
    
@pytest.mark.parametrize(
    "theta1, theta2, expected",
    [
        (  0,   0,          0.0),
        (  0, 0.5, -7.704787325),
        (0.5,   0,  6.768494455),
        (0.5, 0.5,          0.0),
    ],
)
def test_domega2_dt(theta1, theta2, expected):
    dp = DoublePendulum()
    t = 0
    y = (theta1, 0.25, theta2, 0.15)
    _, _, dtheta2_dt, domega2_dt = dp(t, y)
    assert np.isclose(dtheta2_dt, 0.15)
    assert np.isclose(domega2_dt, expected)

def test_pendulum_solution_not_deg_or_rad():
    with pytest.raises(TypeError):
        test3=DoublePendulum()
        test3.solve((60, 9, 60, 9),1,0.01,angle="dreg")

def test_for_move_after_zero():
    test4=DoublePendulum()
    test4.solve((0, 9, 0, 9),1,0.01)
    assert test4.theta1[0] != test4.theta1[1]
    assert test4.theta2[0] != test4.theta2[1]

def test_for_initial_values():
    test5=DoublePendulum()
    test5.solve((0, 0, 0, 0),1,0.01)
    assert (test5.theta1 == np.zeros_like(test5.theta1)).all()  
    assert (test5.theta2 == np.zeros_like(test5.theta2)).all()
    #zeros_like returns an array of the same shape and type of (test5.theta1/theta2) full of zeros
    #.all() is to compare all values 