
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from core_lib.core_fdm import SDEModel, EulerMaruyamaMonteCarlo, MonteCarloSummary, SensitivitySDEAnalysisTool, Calibrator

# A simple forecast with Geometric Brownian Motion for AAPL

data_inv = pd.read_csv('AAPL_250d.csv').iloc[:,1]

temp = data_inv.iloc[::-1]
calibrator_data = temp[:125] # Data from day 0 to day 124
real_data = temp[124:] # Data from day 124 to 250

GBM_cal = Calibrator(1)
parameters = GBM_cal.gbm_calibrate(calibrator_data)

GBM_model = SDEModel(lambda t,x,p : p[0]*x, lambda t,x,p : p[1]*x)
simulator = EulerMaruyamaMonteCarlo(GBM_model, 1, 125)

time, values = simulator.simulate(125,float(calibrator_data.iloc[-1])
                                  ,100000, parameters, 342809)
summarizer = MonteCarloSummary(time,values)
perturber = SensitivitySDEAnalysisTool(simulator, 0 ,0)

print('VaR95', summarizer.time_series_summary()['VaR95'])
print('CVaR95', summarizer.time_series_summary()['CVaR95'])
fig1, ax1 = plt.subplots()
for i in range(100):
    if i ==1 :
        ax1.plot(time,values[i,:],'k', label='Simulated GBM paths')
    else:
        ax1.plot(time,values[i,:],'k')
ax1.plot(np.arange(124,250),np.asarray(real_data), 'b' , label='Actual AAPL prices')
ax1.plot(time, summarizer.time_series_summary()['Mean'], 'g', label='Mean')
ax1.plot(time, summarizer.time_series_summary()['quantiles'][0.05], 'r', label='0.05 quantile')
ax1.set_ylabel('Value (USD)')
ax1.set_xlabel('Time (days)')
ax1.set_title('AAPL GBM forecast')
ax1.legend()
ax1.grid()

fig2, ax2 = plt.subplots()

im = ax2.imshow(perturber.two_param_heatmap(125,float(calibrator_data.iloc[-1]),100
                            ,parameters,0,1,30,2,'VaR95',0.5,'Absolute', 342809) ,extent=(-30*2,30*2,-30*2,30*2), cmap='hot')
ax2.set_title("VaR95 with peturbed parameters")
ax2.set_xlabel("Change in drift (%)")
ax2.set_ylabel("Change in volatility (%)")
fig2.colorbar(im, ax=ax2, label="VaR95")

plt.show()
