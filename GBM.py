
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from core_lib.core_fdm import SDEModel, EulerMaruyamaMonteCarlo, MonteCarloSummary, SensitivitySDEAnalysisTool, Calibrator

# A weekly simple forecast with Geometric Brownian Motion for AAPL

pd.read_csv('FinancialDEModels\AAPL_6M.csv', )