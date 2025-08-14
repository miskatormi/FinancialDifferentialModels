import numpy as np

class SyFunc:
    
    def __init__(self, desc, func):
        self.function = func
        self.desc = desc

    def __str__(self):
        return self.desc + '_SyFunc'


class ODE(SyFunc):
    
    def __init__(self, n : int, F : SyFunc): 
        self.order = n
        self.RHS = F.function
    
    def __str__(self):
        return 'D(Vec[y])=Vec['+str(self.RHS)+']'

class ODE_RK4_Solve(ODE):

    def __init__(self, D : ODE, h: float):
        self.eq = D
        self.tstep = h
    
    def step(self,t0: float ,  x0 : np.array):
        k1 = self.tstep*self.eq.RHS(t0, x0)
        k2 = self.tstep*self.eq.RHS(t0 + self.tstep/2, x0 + k1/2)
        k3 = self.tstep*self.eq.RHS(t0 + self.tstep/2, x0 + k2/2)
        k4 = self.tstep*self.eq.RHS(t0 + self.tstep, x0 + k3)
        return x0 + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
