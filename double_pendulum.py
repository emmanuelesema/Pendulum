import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class DoublePendulum():
    def __init__(self,L1=1, L2=1, g=9.81,M1=1,M2=1):
        self.L1 = L1
        self.L2 = L2
        self.g = g
        self.M1=M1
        self.M2=M2
        self._theta1=None
        self._theta2=None
        M1 = M2 = 1

    def __call__(self, t, y):
        theta1, omega1, theta2, omega2 = y
        g = self.g
        L1 = self.L1
        L2 = self.L2
        dtheta1_dt = omega1
        dtheta2_dt = omega2

        d_t = theta2 - theta1 #delta_theta

        domega1_dt = (L1 * (omega1**2) * np.sin(d_t) * np.cos(d_t) + g * np.sin(theta2) * np.cos(d_t)
        + L2 * (omega2**2) * np.sin(d_t) - 2 * g * np.sin(theta1)) / (2 * L1 - L1 * (np.cos(d_t))**2)

        domega2_dt = (-L2 * (omega2**2) * np.sin(d_t) * np.cos(d_t) + 2 * g * np.sin(theta1) * np.cos(d_t)
        - 2 * L1 * (omega1**2) * np.sin(d_t) - 2 * g * np.sin(theta2)) / (2 * L2 - L2 * (np.cos(d_t))**2)

        return dtheta1_dt, domega1_dt, dtheta2_dt, domega2_dt

    def solve(self,y0,t,dt,angle="rad"):
        self.dt=dt
        if angle=="rad":
            pass
        elif angle=="deg":
            for i in y0:
                i=np.radians(i)
        else:
            raise TypeError("Angle must be rad or deg")
        sol=solve_ivp(self,[0,t],y0=y0,max_step=dt,first_step=dt)
        self._theta1=np.array(sol.y[0]) #chose to return the results
        self._omega1=np.array(sol.y[1]) #as a np array instead of list
        self._theta2=np.array(sol.y[2])
        self._omega2=np.array(sol.y[3])
        self._t=sol.t
#properties for the two thetas and t
    @property
    def theta1(self):
        if self._theta1 is None:
            raise NameError("No solution exists. Use .solve first")
        else:
            return self._theta1

    @property
    def theta2(self):
        if self._theta2 is None:
            raise NameError("No solution exists. Use .solve first")
        else:
            return self._theta2
    @property
    def t(self):
        return self._t
#properties for the coordinates
    @property
    def x1(self):
        return self.L1*np.sin(self.theta1)
    @property
    def y1(self):
        return (-1)*self.L1*np.cos(self.theta1)
    @property
    def x2(self):
        return self.x1+self.L2*np.sin(self.theta2)
    @property
    def y2(self):
        return self.y1-self.L2*np.cos(self.theta2)

#properties for the velocities
    @property
    def vx1(self):
        return np.gradient(self.x1,self.dt)
    @property
    def vx2(self):
        return np.gradient(self.x2,self.dt)
    @property
    def vy1(self):
        return np.gradient(self.y1,self.dt)
    @property
    def vy2(self):
        return np.gradient(self.y2,self.dt)
#properties for the energies
    @property
    def potential(self):
        P1=self.M1*self.g*(self.y1+self.L1)
        P2=self.M2*self.g*(self.y2+self.L1+self.L2)
        return P1+P2
    @property
    def kinetic(self):
        K1=0.5*self.M1*(self.vx1**2+self.vy1**2)
        K2=0.5*self.M2*(self.vx2**2+self.vy2**2)
        return K1+K2

    def create_animation(self):
        # Create empty figure
        fig = plt.figure()
            
        # Configure figure
        plt.axis('equal')
        plt.axis('off')
        plt.axis((-3, 3, -3, 3))
            
        # Make an "empty" plot object to be updated throughout the animation
        self.pendulums, = plt.plot([], [], 'o-', lw=2)
            
        # Call FuncAnimation
        self.animation = animation.FuncAnimation(fig,
                                                 self._next_frame,
                                                 frames=range(len(self.x1)), 
                                                 repeat=None,
                                                 interval=1000*self.dt, 
                                                 blit=True)
    
    def _next_frame(self, i):
            self.pendulums.set_data((0, self.x1[i], self.x2[i]),
                                    (0, self.y1[i], self.y2[i]))
            return self.pendulums,

    def show_animation(self):
        plt.show
    
    def save_animation(self, filename):
        return self.animation.save(filename, fps=60)


if __name__=="__main__":
    test1=DoublePendulum(L1=1,L2=0.5,M1=3,M2=1)
    test1.solve((0.1,0.3,0.5,0.8),10,0.016)
    plt.plot(test1.t, test1.kinetic, label = 'K.E.')
    plt.plot(test1.t, test1.potential, label = 'P.E.')
    plt.plot(test1.t, test1.potential+test1.kinetic, label = 'Total Energy')
    plt.legend()
    plt.show()
    
    #Animation
    test1.create_animation() 
    test1.show_animation()
    #test1.save_animation("pendulum_motion.mp4")