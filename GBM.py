
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from core_lib.core_fdm import SDEModel, EulerMaruyamaMonteCarlo, MonteCarloSummary, SensitivitySDEAnalysisTool, Calibrator

# A weekly simple forecast with Geometric Brownian Motion for AAPL

data_inv = pd.read_csv('AAPL_6M.csv').iloc[:,1]

data = data_inv.iloc[::-1]


GBM_cal = Calibrator(1)
parameters = GBM_cal.gbm_calibrate(data)
print(parameters)