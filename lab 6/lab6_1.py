import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

#==============================================================================
#==================================MA models===================================
#==============================================================================

def AMq_model(end, c, burnin, params, init, eps):
    results = init
    for i in range((len(init)-1), end):
        for j in range(len(params)):
            results.append(c+sum([params[j]*eps[i-j]]) + eps[i])
    return np.array(results[burnin:])

#==============================================================================
#===================================AR Model===================================
#==============================================================================

def ARp_model(end, c, burnin, params, init, eps):
    results = init
    for i in range((len(init)-1), end):
        for j in range(len(params)):
            results.append(c+sum([params[j]*results[i-j]]) + eps[i])
    return np.array(results[burnin:])

#==============================================================================
#=============== n=5000, MA(1), y_t = 20 + e_t + 0,8*e_(t-1) ==================
#==============================================================================

n = 5000
burnin = 200
c = 20
theta = 0.8
noise = np.random.normal(size=(n+burnin))

df1 = AMq_model(n+burnin, c, burnin, [theta], [c], noise)

plt.plot(df1)
plot_acf(df1, lags=30, zero=False)

fig, axes = plt.subplots(1,3, figsize=(18,6))
axes[0].plot(df1)
axes[0].set_title(f'Time series Theta={theta} and c={c}')
axes[1].plot(acf(df1)[1:], marker='o')
axes[1].set_title('ACF')
axes[2].plot(pacf(df1)[1:], marker='o')
axes[2].set_title('Pacf')

#==============================================================================
#=========== n=5000, MA(2), y_t = e_t - e_(t-1) + 0.8*e_(t-2) =================
#==============================================================================

n = 5000
burnin = 200
c = 0
theta = [-1, 0.8]
noise = np.random.normal(size=(n+burnin))

df2 = AMq_model(n+burnin, c, burnin, theta, [c, c], noise)

#plt.plot(df2)
#plot_acf(df2, lags=30, zero=False)

fig, axes = plt.subplots(1,3, figsize=(18,6))
axes[0].plot(df2)
axes[0].set_title(f'Time series Theta={theta} and c={c}')
axes[1].plot(acf(df2)[1:], marker='o')
axes[1].set_title('ACF')
axes[2].plot(pacf(df2)[1:], marker='o')
axes[2].set_title('Pacf')

#==============================================================================
#=========== n=5000, RM(2), y_t = e_t - y_(t-1) + 0.8*y_(t-2) =================
#==============================================================================

n = 5000
burnin = 200
c = 0
theta = [-1, 0.8]
noise = np.random.normal(size=(n+burnin))

df3 = ARp_model(n+burnin, c, burnin, theta, [c], noise)

#plt.plot(df3)
#plot_acf(df3, lags=30, zero=False)

fig, axes = plt.subplots(1,3, figsize=(18,6))
axes[0].plot(df3)
axes[0].set_title(f'Time series Theta={theta} and c={c}')
axes[1].plot(acf(df3)[1:], marker='o')
axes[1].set_title('ACF')
axes[2].plot(pacf(df3)[1:], marker='o')
axes[2].set_title('Pacf')
