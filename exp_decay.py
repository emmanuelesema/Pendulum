import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

"""Class To Model Exponential Decay of a Substance"""
class ExponentialDecay:
    def __init__(self,a):
        """Function to Define Parameters"""
        if a<0:
            raise ValueError("a cannot be a negative number")
        else:
             self.a=a

    def __call__(self, t, u):
        """Function to Call t, u values"""
        self.u = float(u)
        self.a = a
        dudt = u * -a
        return round(dudt,3)

    def solve(u0, T, dt):
        """Function to Solve ODE"""
        t = np.linspace(0, 10, 300)
        u0 = [3.2]
        sol = solve_ivp(lambda t, u: -a*u,
                        [0,10], 
                                u0, 
                                    t_eval=t, 
                                        rtol = 1e-5)
        t = sol.t
        u = sol.y[0]
        return t, u

T = 10
dt = 300
a=0.4

decay_model = ExponentialDecay(a)
t, u = decay_model.solve(T, dt)

#print(f't-values: {decay_model.solve(T, dt)[0]}')
#print(f'u-values: {decay_model.solve(T, dt)[1]}')

plt.plot(t, u)
plt.xlabel('t values')
plt.ylabel('u values')
plt.show()