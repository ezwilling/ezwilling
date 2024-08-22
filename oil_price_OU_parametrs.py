#%%
# -*- coding: utf-8 -*-
"""
File name: oil_price_OU_parametrs.py
Created by Zvika Afik on Sat Jun 22 11:12:23 2024
@author: AfikZv
Description: This file was used to develop Vasicek_functions.py and
provides examples for the use of the functions
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm 
from Vasicek_functions import * # special purpose function set
os.chdir("/Users/eranzwilling/Documents/Doctorate/HedgeOil/code/zvika/")
data_file = "crude_oil_real_monthly_prices.xls"
# data_file = "crude_oil_RWTC.xls" # nominal daily prices
df = pd.read_excel(data_file)
# r = df.pct_change(1) # no need, we assume price follows OU process

'Estimate a, b, and sigma using linear regression'

LS_Estimate = Vasicek_LS(df.to_numpy(), 1/12)
print("a_est: " + str(np.round(LS_Estimate[0],3)))
print("b_est: " + str(np.round(LS_Estimate[1],3)))
print("sigma_est: " + str(np.round(LS_Estimate[2],3)))


p0 = df.iloc[0]  # Initial price
num_steps = 605  # Number of steps
T = num_steps/12     # Time horizon
num_paths = 10   # Number of paths
a = LS_Estimate[0]
b = LS_Estimate[1]
sigma = LS_Estimate[2]

# Simulate Vasicek model using Euler-Maruyama Discretisation
simulated_prices = vasicek_EM(p0, a, b, sigma, T, num_steps, num_paths)
time_axis = np.linspace(0, T, num_steps + 1)

# Plotting multiple paths with time on x-axis
plt.figure(figsize=(10, 6))
plt.title('OU Model - simulated crude Oil paths using EM discretization', fontsize=18)
plt.xlabel('Time (years)', fontsize=16)
plt.ylabel('Real Price [$/Barrel]', fontsize=16)
for i in range(num_paths):
    plt.plot(time_axis, simulated_prices[:, i])
plt.plot(time_axis, df.to_numpy(), '-k', linewidth=4)



# Simulate Vasicek model using exact discretization
simulated_prices_exact = vasicek_ZA(p0, a, b, sigma, T, num_steps, num_paths)

# Plotting multiple paths with time on x-axis
plt.figure(figsize=(10, 6))
plt.title('OU Model - simulated crude Oil paths using exact discretization', fontsize=16)
plt.xlabel('Time (years)', fontsize=16)
plt.ylabel('Real Price [$/Barrel]', fontsize=16)
for i in range(num_paths):
    plt.plot(time_axis, simulated_prices_exact[:, i])
plt.plot(time_axis, df.to_numpy(), '-k', linewidth=4)


"""
Repeat OU parameter estimation, now using MLE
It seems that MLE results are very similar to those of OLS 
"""
MLE_Estimate = Vasicek_MLE(df.to_numpy(), 1/12)
print("a_est: " + str(np.round(MLE_Estimate[0],3)))
print("b_est: " + str(np.round(MLE_Estimate[1],3)))
print("sigma_est: " + str(np.round(MLE_Estimate[2],3)))

# The following code is an attempt to use qq-plot for the OU case
# y = df.to_numpy()
# x = np.concatenate(simulated_prices)
# QQ_ZA(x, y, "simulated price", "real price", "QQ plot")


