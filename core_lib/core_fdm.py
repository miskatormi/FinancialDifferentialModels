import numpy as np
import pandas as pd
import warnings

class ODEModel:
    """

    A class modelling an Ordinary Differential Equation of the form D_t(x)=RHS(t, x, params=...). Note here that RHS and x must have same shape!

    ********

    Parameters:

    F : callable (function) (RHS meaning the right hand side of the ODE)

    ********

    """
    def __init__(self, F):
        self.RHS = F
    
    def __str__(self):
        return 'dx/dt='+str(self.RHS)

class RK4Solve:
    """ 
    
    Runge-Kutta 4 Solver for the ODEModel class:
    
    ********
    
    Parameters:
    
    D : ODEModel
    an ODE model

    h : float
    size of the time step used in simulation

    n : int
    total number of steps
    
    ********
    
    Methods:
    next_step(t : float, x : np.ndarray, params : np.ndarray) -> np.ndarray : 
    Takes initial values x and time t to give the value x at time t + self.tstep.

    solve(t : float, x : np.ndarray, paramsarr : np.ndarray) -> Tuple[np.ndarray , np.ndarray (with shape: (state_dim, n_steps+1))]  : 
    Takes initial values x and time t to give the full simulated data of all values x at all times t. (paramsarr.shape == (param_dim, n_steps))

    ********

    """

    def __init__(self, D : ODEModel, h: float, n : int):
        self.eq = D
        self.tstep = float(h)
        self.nsteps = int(n)
    
    def next_step(self,t: float ,  x : np.ndarray, params : np.ndarray = None): # Returns reliable results for nicely behaving parameters.
        if params is None:
            k1 = self.tstep*self.eq.RHS(t, x)
            k2 = self.tstep*self.eq.RHS(t + self.tstep/2, x + k1/2)
            k3 = self.tstep*self.eq.RHS(t + self.tstep/2, x + k2/2)
            k4 = self.tstep*self.eq.RHS(t + self.tstep, x + k3)
        else:
            k1 = self.tstep*self.eq.RHS(t, x, params)
            k2 = self.tstep*self.eq.RHS(t + self.tstep/2, x + k1/2, params)
            k3 = self.tstep*self.eq.RHS(t + self.tstep/2, x + k2/2, params)
            k4 = self.tstep*self.eq.RHS(t + self.tstep, x + k3, params)
        return x + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
    
    def solve(self, t : float, x : np.ndarray, paramsarr : np.ndarray = None): 
        x = np.atleast_1d(x).astype(float)
        n1x = np.shape(x)[0]
        time = np.zeros(self.nsteps+1,dtype=float)
        value = np.zeros((n1x,self.nsteps+1),dtype=float)
        value[:,0] = x
        time[0] = t
        if paramsarr is None:
            for i in range(1,self.nsteps+1):
                value[:,i] = self.next_step(time[i-1], value[:,i-1])
                time[i] = t + i*self.tstep
        elif paramsarr.ndim == 1:
            correct_p = paramsarr[:,np.newaxis]*np.ones((1,self.nsteps+1))
            for i in range(1,self.nsteps+1):
                value[:,i] = self.next_step(time[i-1], value[:,i-1], correct_p[:,i-1])
                time[i] = t + i*self.tstep
        else:
            for i in range(1,self.nsteps+1):
                value[:,i] = self.next_step(time[i-1], value[:,i-1], paramsarr[:,i-1])
                time[i] = t + i*self.tstep
        return time, value
        


    
class SDEModel:

    """
    
    A class modelling a Stochastic Differential Equation of the form dX_t=a*dt+b*dW_t, where dW_t is normally distributed.

    ********

    Parameters:

    a : callable (function)

    b : callable (function)

    ********

    """

    def __init__(self, a, b):
        self.func_dt = a
        self.func_dW = b

    def __str__(self):
        return 'dX_t= ' + str(self.func_dt) + '*dt + ' + str(self.func_dW) + '*dW_t'
    
class EulerMaruyamaMonteCarlo:

    """

    A simulator for a SDE using the Euler-Maruyama algorithm and the Monte Carlo method.

    ********

    Parameters:

    Model : SDEModel
    The SDE we want to solve

    h : float
    Size of the time step

    n : int
    The total number of time steps in the simulation

    ********

    Methods:

    next_step_monte_carlo(t : float, x: np.ndarray,rn , params : np.ndarray = None) -> np.ndarray : 
    Takes in an array vector x filled with the starting value, the starting time t
    and the parameters array params and return vector of the same shape as x but after
    time self.tstep, each component is a different path of the stochastic process.

    simulate(t : float, x0 : float, npaths : int , paramsarr : np.ndarray = None, rng = None) -> Tuple[np.ndarray,np.ndarray (with shape: (paths, n_steps+1))] : 
    Gives all the steps from intial value x and inital time t of the npaths number of different paths
    and additionally returns the corresponding times t. (paramsarr.shape == (param_dim, n_steps))

    ********

    """

    def __init__(self, Model : SDEModel, time_step_size : float, number_of_time_steps : int):
        self.model = Model
        self.tstep = float(time_step_size)
        self.nsteps = int(number_of_time_steps)
    
    def next_step_monte_carlo(self, t : float, x : np.ndarray, rn : np.random.Generator, params : np.ndarray = None):
        dW_t = rn.normal(0, np.sqrt(self.tstep), np.shape(x))
        if params is None:
            return x + self.model.func_dt(t,x)*self.tstep + self.model.func_dW(t,x)*dW_t
        else:
            return x + self.model.func_dt(t,x, params)*self.tstep + self.model.func_dW(t,x, params)*dW_t
    
    def simulate(self, t : float, x0 : float, npaths: int, paramsarr : np.ndarray = None, rng = None):
        if rng is None:
            rn = np.random.default_rng()
        elif isinstance(rng, int):
            rn = np.random.default_rng(rng)
        elif isinstance(rng, np.random.Generator):
            rn = rng
        else:
            raise ValueError("rng must be None, int, or np.random.Generator")

        x = np.full(npaths,x0)
        time = np.zeros(self.nsteps+1,dtype=float)
        value = np.zeros((np.shape(x)[0],self.nsteps+1),dtype=float)
        value[:,0] = x
        time[0] = t
        if paramsarr is None:
            for i in range(1,self.nsteps+1):
                value[:,i] = self.next_step_monte_carlo(time[i-1], value[:,i-1],rn)
                time[i] = t + i*self.tstep
        elif paramsarr.ndim == 1:
            correct_p = paramsarr[:,np.newaxis]*np.ones((1,self.nsteps+1),dtype=float)
            for i in range(1,self.nsteps+1):
                value[:,i] = self.next_step_monte_carlo(time[i-1], value[:,i-1],rn, params=correct_p[:,i-1])
                time[i] = t + i*self.tstep
        else:
            for i in range(1,self.nsteps+1):
                value[:,i] = self.next_step_monte_carlo(time[i-1], value[:,i-1],rn, params=paramsarr[:,i-1])
                time[i] = t + i*self.tstep
        return time, value
    

    
class MonteCarloSummary:
    """
    
    A class meant to model a data structure which has many stochastic walks in some time interval, to tell statistics of all paths like how the standard deviation developed with respect to time.

    ********

    Parameters:
    
    times : np.ndarray
    The time data points on the wanted interval.

    value : np.ndarray
    The row vectors contain the different walks/paths of the stochastic process.

    ********

    Methods:

    time_series_summary() -> dict
    Calculates most statistics for each time step of the different paths. Also returns VaR95, CVaR95 statistics.

    terminal_value() -> np.ndarray
    Terminal values across paths. 

    ********

    """

    def __init__(self, times : np.ndarray, value : np.ndarray):
        self.times = times
        self.value = value

    def time_series_summary(self):
        if self.value[0,0] == 0:
            VaR95 = np.nan
            CVaR95 = np.nan
            warnings.warn('Starting value cannot be zero, VaR95/CVaR95 undefined')
        else:
            TerminalReturn = self.value[:,-1]/self.value[0,0]-1
            TerminalLoss = - TerminalReturn
            VaR95 = np.quantile(TerminalLoss, 0.95)
            mask = TerminalLoss >= VaR95
            CVaR95 = np.mean(TerminalLoss[mask])
        return {'time': self.times,
                'Mean': np.mean(self.value, axis=0),
                'Median': np.median(self.value, axis=0),
                'StandardDeviation': np.std(self.value, axis=0),
                'Max':np.max(self.value, axis=0),
                'Min':np.min(self.value, axis=0),
                'quantiles': {q : np.quantile(self.value,q, axis=0) for q in [0.05,0.25,0.50,0.75,0.95]},
                'VaR95' : VaR95,
                'CVaR95' : CVaR95
                }
    
    def terminal_value(self):
        return self.value[:,-1]

class SensitivitySDEAnalysisTool():
    """
    
    A class that models a simple sensitivity analysis tool.

    ********

    Parameters:

    SModel : EulerMaruyamaMonteCarlo
    The solver of the SDE we are inspecting

    percentage : float
    How much you want to add to the paramater

    paramspecifier : int
    Which parameter you want to change

    ********

    Methods:

    tell_new_result(t : float, x : float, npaths : int, paramsarr : np.ndarray) -> Tuple[np.ndarray, np.ndarray]
    Returns the result of the simulation using the new parameters. t,x are the starting time and value. 
    npaths is the number of paths of the stochastic process that we want to calculate. paramsarr is the
    array which contains the parameters.

    tell_new_result_summaries(t : float, x : float, npaths : int , paramsarr : np.ndarray) -> Tuple[np.ndarray, dict]
    Returns all possible statistics using the MonteCarloSummary class of the new simulated values.t,x are the starting time and value. 
    npaths is the number of paths of the stochastic process that we want to calculate. paramsarr is the array which contains the parameters.

    two_param_heatmap(t : float, x : float, npaths : int, paramsarr : np.ndarray, idx1 : int, idx2 : int, nsteps : int, step_percent : float, value_rep : str, quantile : float, param_rep : str, rng = None) -> np.ndarray
    Returns a np.ndarray of shape (2*nsteps+1,2*nsteps+1) where each value depicts a pixel and its intensity. In each cell the asked value_rep is
    simulated with differently perturbed parameters 1 and 2. These parameters are in the parameter array:
    
    parameter 1 = paramsarr[idx1]
    parameter 2 = paramsarr[idx2]

    idx1, idx2 determine which two parameters are perturbed. The perturbation is done so that in the center cell the parameters are not changed.
    and when moving away from the center cell the parameters are perturbed always by step_percent.

    ********

    """
    def __init__(self, SModel : EulerMaruyamaMonteCarlo, percentage : float, paramspecifier : int):
        self.SolModel = SModel
        self.coeff = percentage/100+1
        self.param = int(paramspecifier)

    def tell_new_result(self, t : float, x : float, npaths : int , paramsarr : np.ndarray):
        new_params = paramsarr.copy()
        new_params[self.param] *= self.coeff
        return self.SolModel.simulate(t,x,npaths,new_params)
    
    def tell_new_result_summaries(self, t : float, x : float, npaths : int, paramsarr : np.ndarray):
        new_params = paramsarr.copy()
        new_params[self.param] *= self.coeff
        time, value = self.SolModel.simulate(t,x,npaths,new_params)
        summaryNew = MonteCarloSummary(time,value)
        return summaryNew.terminal_value(), summaryNew.time_series_summary()
    
    def two_param_heatmap(self, t : float, x : float, npaths :int , paramsarr : np.ndarray, idx1 : int, idx2 : int , nsteps : int, step_percent : float, value_rep : str, quantile : float , param_rep : str, rng = None):
        if rng is None:
            rn = None
        elif isinstance(rng, int):
            rn = np.random.default_rng(rng)
        elif isinstance(rng, np.random.Generator):
            rn = rng
        else:
            raise ValueError("rng must be None, int, or np.random.Generator")

        def compute_cell(i,j):
            n_params = paramsarr.copy()
            n_params[idx1] = (1+i*step_percent/100)*paramsarr[idx1]
            n_params[idx2] = (1+j*step_percent/100)*paramsarr[idx2] 
            time, value = self.SolModel.simulate(t,x,npaths,n_params, rng=rn)
            ts = MonteCarloSummary(time,value).time_series_summary()
            if value_rep not in ts:
                raise ValueError('Invalid value for value_rep.')
            d = ts[value_rep]
            if value_rep == 'quantiles':
                if quantile not in ts[value_rep]:
                    raise ValueError('Invalid value for quantile')
                else:
                    return d[quantile][-1]
            if np.isscalar(d):
                if np.isnan(d):
                    raise ValueError('Value for a cell of the heatmap could not be calculated. Possibly because of a bad starting value.')
                return d
            else:
                return d[-1]

        
        res = np.zeros((nsteps*2+1,nsteps*2+1), dtype=float)

        for i in range(-nsteps,nsteps+1):
            for j in range(-nsteps,nsteps+1):
                res[i+nsteps,j+nsteps] = compute_cell(i,j)


        if param_rep == 'Absolute':
            return res
        elif param_rep == 'Percent':
            if res[nsteps,nsteps]==0:
                raise ValueError('The base value was zero, preventing zero division')
            return (res - res[nsteps,nsteps])/res[nsteps,nsteps]*100
        else:
            raise ValueError('No acceptable value for param_rep was given.')

class Calibrator():
    """

    A Calibrator to calibrate GBM and Vasicek SDE models.

    ********

    Parameters:

    time_interval : float
    Tells the time interval between datapoints.

    ********

    Methods:

    gbm_calibrate(data : pd.Series) -> np.ndarray
    Takes in data and returns suitable drift (m) and diffusion (sigma) coefficients for geometric brownian
    motion in a np.ndarray np.array([m,sigma]). The method calculates these using the log-returns.

    vasicek_calibrate(data : pd.Series) -> np.ndarray
    Takes in data and returns suitable speed of mean reversion (k), long-run mean (t) and diffusion (s) coefficients
    for the Vasicek SDE model in a np.ndarray  np.array([k,t,s]). It uses least-squares method to determine the relationship
    between the shifted data and the data i.e the coefficients k ,t (fitting : data_{i+1}=c+d*data_i+residue_i). However s is determined
    from the residues of each datapoint.

    ********

    """

    def __init__(self, time_interval : float):
        if time_interval <= 0:
            raise ValueError('Time interval cannot be zero or negative!')
        self.tint = float(time_interval)

    def gbm_calibrate(self, data : pd.Series):
        if len(data.dropna()) < 3 : 
            raise ValueError('Dataset too short.')
        data_n=np.asarray(data.dropna(), dtype=float)
        if np.any(data_n <= 0):
            raise ValueError('A data value was negative or zero.')
        log_dat = np.log(data_n[1:]/data_n[:-1])
        sigma = np.sqrt(log_dat.var()/self.tint)
        mu = log_dat.mean() / self.tint + 0.5*sigma**2
        return np.array([ mu, sigma], dtype=float)
    
    def vasicek_calibrate(self, data : pd.Series):
        if len(data.dropna()) < 3 : 
            raise ValueError('Dataset too short.')
        data_n = np.asarray(data.dropna(), dtype=float)
        final_data = data_n[:-1]
        shifted_final_data = data_n[1:]
        c, d =np.linalg.lstsq(np.column_stack([np.ones_like(final_data),final_data]), shifted_final_data, rcond=None)[0]
        if d < 1e-6 or d > 0.999:
            warnings.warn('Nearly unsuitable coefficient, clipping may be done.')
        d_safe = float(np.clip(d,1e-8,1-1e-8))
        eps = shifted_final_data - c - d_safe * final_data
        k = -np.log(d_safe)/self.tint
        t = c/(1-d_safe)
        s = np.sqrt(2*k*np.mean(eps**2)/(1-d_safe**2))
        return np.array([k, t, s] ,dtype=float)

