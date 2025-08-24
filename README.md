# FinancialDifferentialModels
Numerical simulation of deterministic and stochastic financial models using 4th order Runge-Kutta, Euler-Maruyama and Monte Carlo methods.
## Motivation
This repository showcases my interest in financial modeling, data analysis, programming, and mathematics. While studying physics, I developed a strong fascination with the mathematical and computational aspects of financial markets. This project represents a selection of my personal work exploring such models.
## Core Solver **`Core_fdm`**
### Features
#### ODE simulation

**`ODEModel`** : 

Encapsulates the deterministic ODE model by saving the right hand side of the ODE, that should be in the form 

$$\vec{\dot x}=\vec f(t,x,\text{params})$$

So the class saves the function $f$ and assumes the ODE is in the above form.

**`RK4Solver`** : 

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

#### SDE simulation

**`SDEModel`** :

Saves drift and diffusion terms of a SDE model. Specifically it saves the functions a,b from the stochastic process:

$$ dX_t=a(X_t,t, \text{params})dt+b(X_t,t, \text{params})dW_t $$

**`EulerMaruyamaMonteCarlo`** :


    

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



#### Analysis and summary tools

**`MonteCarloSummary`** :

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

**`SensitivitySDEAnalysisTool`** :

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

**`Calibrator`** :

    A Calibrator to calibrate GBM and Vasicek SDE models from historical data.

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

## Examples of using the Core Solver
### Deterministic Models
#### Single Asset Model

<iframe
  src="https://https://github.com/miskatormi/FinancialDifferentialModels/blob/main/figures/single_asset_model.pdf"
  width="100%"
  height="600px"
  style="border: none;">
</iframe>

#### Multi Asset Model
### Stochastic Models (Monte Carlo simulations)
#### Geometric Brownian Motion for AAPL + VaR95 heatmap for perturbed drift and volatility to evaluate risk
#### Vasicek SDE for FEDFUNDS for risk evaluation

