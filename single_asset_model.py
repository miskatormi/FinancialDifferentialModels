
import matplotlib.pyplot as plt
import numpy as np
from core_lib.core_fdm import ODEModel,RK4Solve

# Time-varying rate, an asset with seasonality in its growth rate:
base_growth = 0.05
oscillatory_growth_amplitude = 0.07
period_of_oscillation = 1
initial_value = 100

starting_time = 0

parameters = np.array([base_growth, oscillatory_growth_amplitude, period_of_oscillation])

single_asset_timev = ODEModel(lambda t,x,p: (p[0]+p[1]*np.sin(2*np.pi*t/p[2]))*x)
RK4Solver = RK4Solve(single_asset_timev, 0.01, 1000)
time, value = RK4Solver.solve(starting_time, initial_value, parameters)

analytical_value = initial_value*np.exp(base_growth*time + (oscillatory_growth_amplitude*period_of_oscillation/(2*np.pi))
                                        *(1-np.cos(2*np.pi*time/period_of_oscillation)))

difference = np.abs(analytical_value-value[0,:])

fig, ax = plt.subplots(1,2)

ax[0].plot(time,value[0,:], 'b')
ax[0].set_xlabel('Time (years)')
ax[0].set_ylabel('Value (€)')
ax[0].set_title('Numerical solution of single asset model')
ax[0].grid()

ax[1].plot(time,difference, 'r')
ax[1].set_xlabel('Time (years)')
ax[1].set_ylabel('Difference of solutions')
ax[1].set_title('Difference of numerical and analytical solution')
ax[1].grid()

plt.show()

