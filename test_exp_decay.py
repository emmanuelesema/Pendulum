from exp_decay import *
import unittest

def test_exponentialdecay():
    """Test Veryfying That ExponentialDecay Raises ValueError For Negative values"""    
    try:
        ExponentialDecay(-1)
    except ValueError:
        pass

    """Test Veryfying That ExponentialDecay Works"""
    x = ExponentialDecay(0.4)   #create an instance
    assert x(0,3.2) == -1.28    #__call__ instance

    """Test Veryfying Exact Value == ODE Value"""
    def f():
        t = np.linspace(0, 10, 300)
        u0 = 3.2
        u = u0*np.exp(-a*t)
        return u
    np.logical_and(decay_model.solve(T, dt)[1], f())