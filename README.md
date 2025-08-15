# FinancialDEModels
Numerical simulation of deterministic and stochastic financial models using 4th order Runge-Kutta, Euler-Maruyama and Monte Carlo methods.
## Motivation
This repository showcases my interest in financial modeling, data analysis, programming, and mathematics. While studying physics, I developed a strong fascination with the mathematical and computational aspects of financial markets. This project represents a selection of my personal work exploring such models.
## Core Solver
### Features
**ODE simulation:**

**`ODEModel`** : 

Encapsulates the deterministic ODE model by saving the right hand side of the ODE, that should be in the form 

$$\vec{\dot x}=\vec f(t,x,params)$$

So the class saves the function $f$ and assumes the ODE is in the above form.

**`RK4Solver`** : 

Implements the Runge-Kutta 4 algorithm to solve the given ODEModel for a specific number of time steps of specific size.

**SDE simulation** : 

**`SDEModel`** :

Saves drift and diffusion terms of a SDE model.

**`EulerMaruyamaMonteCarlo`** :

Simulates a given number of paths of the stochastic process form a given starting point.


### Analysis tools
#### Sensitivity analysis
#### Monte Carlo summary


## Deterministic Models
### Single Asset Model
### Multi Asset Model
### Feedback Rate Growth Models
#### Diminishing returns
#### Cost-of-debt Feedback
## Stochastic Models (Monte Carlo simulations)
### Geometric Brownian Motion, GBM
#### Constant drift and volatility
#### Time Dependent drift and volatility
### Vasicek SDE
### DCF with Vasicek

## Data-driven simulations
### Estimation of drift and volatility for GBM from market price data
### Estimation of long-run mean & volatility for Vasicek interest rate models
### Calibration from historical datasets to make forward-looking simulations
