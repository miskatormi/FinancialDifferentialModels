
import matplotlib.pyplot as plt
import numpy as np
from core_lib.core_fdm import ODEModel,RK4Solve

# Captial competition, two sectors compete for scarce funding

capital_competition = ODEModel(lambda t,x,p : np.array([x[0]*(p[0]-p[1]*x[0]-p[2]*x[1])
                                                        ,x[1]*(p[3]-p[4]*x[1]-p[5]*x[0])]))

a1 = 0.08 # Base growth
a2 = 0.05 

b1 = 1e-3 # self-limitation
b2 = b1

c12 = 6e-4 # cross-competition coefficient
c21 = 8e-4

initial_funding1 = 120 
initial_funding2 = 100


parameters = np.array([a1,b1,c12,a2,b2,c21])