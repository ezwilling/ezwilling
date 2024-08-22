
# # -*- coding: utf-8 -*-
"""
File name: Vasicek_functions.py
Created by Zvika Afik on Sat Jun 22 10:46:32 2024
@author: AfikZv
Description: a collection of functions for discrete simulation of
Vasicek (OU) processes and their parameter estimation.

Brigo, D., Dalessandro, A., Neugebauer, M., & Triki, F. (2009). 
  A stochastic processes toolkit for risk management: 
  Mean reverting processes and jumps. 
  Journal of Risk management in Financial institutions, 3(1).
  
https://quant-next.com/the-vasicek-model/

An excellent review of methods and matters is in
https://hudsonthames.org/caveats-in-calibrating-the-ou-process/
"""

import math
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot_2samples

#------------------------------------------------------------------
#  Vasicek Process Simulation using Exact Discretisation
#------------------------------------------------------------------
def vasicek_ZA(r0, a, b, sigma, T, num_steps, num_paths):
    """
    Vasicek Process Simulation using Exact Discretisation
    r0 is the initial value
    a is the speed of mean reversion
    b is the long-term mean
    sigma is the instantaneous volatility
    EZ: 
        T is the time horizon
        num_steps is the number of steps
        num_paths is the number of path or realisations
    
    """
    dt = T / num_steps
    rates = np.zeros((num_steps + 1, num_paths))
    rates[0] = r0
    ex_dt = np.exp(-a*dt)
    for t in range(1, num_steps + 1):
        dW = np.random.normal(0, 1, num_paths)
        rates[t] = ex_dt * rates[t - 1] + b*(1 - ex_dt) + sigma * np.sqrt((1-ex_dt**2)/(2*a)) * dW
    
    return rates

#------------------------------------------------------------------
#  Vasicek Process Simulation using Euler-Maruyama Discretisation
#------------------------------------------------------------------
def vasicek_EM(r0, a, b, sigma, T, num_steps, num_paths):
    """
    Aproximate Vasicek Process Simulation using Euler-Maruyama 
    Discretisation
    r0 is the initial value
    a is the speed of mean reversion
    b is the long-term mean
    sigma is the instantaneous volatility
    
    """
    dt = T / num_steps
    rates = np.zeros((num_steps + 1, num_paths))
    rates[0] = r0
    
    for t in range(1, num_steps + 1):
        dW = np.random.normal(0, 1, num_paths)
        rates[t] = rates[t - 1] + a * (b - rates[t - 1]) * dt + sigma * np.sqrt(dt) * dW
    
    return rates

#----------------------------------------
#      Linear Regression Estimation
#----------------------------------------
def Vasicek_LS(r, dt):
    """
    Vasicek_LS(r, dt) returns a, b, and sigma OLS parameter estimaiton
    of 
    r is the sample OU process with time increments dt 
    
    While b and sigma are usually well estimated, a is known 
    to be positively biased and it remains a challenge """
    
    #Linear Regression
    r0 = r[:-1,]
    r1 = r[1:, 0]
    reg = LinearRegression().fit(r0, r1)
    
    #estimation a and b
    a_LS = (1 - reg.coef_) / dt
    b_LS = reg.intercept_ / dt / a_LS
    
    #estimation sigma
    epsilon = r[1:, 0] - r[:-1,0] * reg.coef_
    sigma_LS = np.std(epsilon) / dt**.5

    return a_LS[0], b_LS[0], sigma_LS


#----------------------------------------
#      Maximum Likelihood Estimation
#----------------------------------------
def Vasicek_MLE(r, dt):
    """
    Vasicek_LS(r, dt) returns a, b, and sigma MLE parameter estimaiton
    of 
    r is the sample OU process with time increments dt 
    
    While b and sigma are usually well estimated, a is known 
    to be positively biased and it remains a challenge """
    r = r[:, 0]
    n = len(r)
    #estimation a and b
    S0 = 0
    S1 = 0
    S00 = 0
    S01 = 0
    for i in range(n-1):
        S0 = S0 + r[i]
        S1 = S1 + r[i + 1]
        S00 = S00 + r[i] * r[i]
        S01 = S01 + r[i] * r[i + 1]
    S0 = S0 / (n-1)
    S1 = S1 / (n-1)
    S00 = S00 / (n-1)
    S01 = S01 / (n-1)
    b_MLE = (S1 * S00 - S0 * S01) / (S0 * S1 - S0**2 - S01 + S00)
    a_MLE = 1 / dt * np.log((S0 - b_MLE) / (S1 - b_MLE))
    
    #estimation sigma
    beta = 1 / a_MLE * (1 - np.exp(-a_MLE * dt))
    temp = 0
    for i in range(n-1):
        mi = b_MLE * a_MLE * beta + r[i] * (1 - a_MLE * beta)
        temp = temp + (r[i+1] - mi)**2
    sigma_MLE = (1 / ((n - 1) * beta * (1 - .5 * a_MLE * beta)) * temp)**.5
    return a_MLE, b_MLE, sigma_MLE

#----------------------------------------
#      2-sample Q-Q Plot
#----------------------------------------
def QQ_ZA(x, y, x_text, y_text, QQ_title):
    """
    QQ_ZA should generate a 2-sample qq-plot accepting
    sample of different lengths. HOWEVER it doesn't seem to work well
    """
    pp_x = sm.ProbPlot(x)
    pp_y = sm.ProbPlot(y)
    # plt.figure(figsize=(10, 6))
    qqplot_2samples(pp_x, pp_y)
    
    plt.title(QQ_title)
    plt.xlabel(x_text)
    plt.ylabel(y_text)
# %%
