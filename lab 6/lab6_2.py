#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss

from statsmodels.tsa.arima.model import ARIMA

from statsmodels.tsa.arima_process import arma_generate_sample

#%%

df=pd.read_csv('INVCMRMT.csv', index_col='DATE', parse_dates=True) 
df.index.freq = 'MS'

print(df.head(10))

#%%

plt.plot(df)

#%%

fig = plot_acf(df, lags=20, zero=False)

#%%

fig = plot_pacf(df, lags=20, zero=False)

#%%

df1 = df.diff().dropna()
print(df1.head(10))

#%%

plt.plot(df1)

#%%

fig = plot_acf(df1, lags=20, zero=False)

#%%

fig = plot_pacf(df1, lags=20, zero=False)

#%%
model1 = ARIMA(df, order=(1,0,0)).fit()
print(model1.summary())

#%%
model2 = ARIMA(df, order=(2,0,0)).fit()
print(model2.summary())

#%%
model3 = ARIMA(df, order=(3,0,1)).fit()
print(model3.summary())

#%%
model4 = ARIMA(df, order=(4,0,1)).fit()
print(model4.summary())

#%%
model1_1 = ARIMA(df, order=(1,0,1)).fit()
print(model1_1.summary())

#%%
model2_1 = ARIMA(df, order=(2,0,1)).fit()
print(model2_1.summary())

#%%
model3_1 = ARIMA(df, order=(3,0,1)).fit()
print(model3_1.summary())

#%%
model4_1 = ARIMA(df, order=(4,0,1)).fit()
print(model4_1.summary())

#%%

model2_2 = ARIMA(df, order=(2,0,2)).fit()
print(model2_2.summary())
#%%
from scipy.stats.distributions import chi2

def LLR_test1(m1,m2,DF=1):
    L1 = m1.llf
    L2 = m2.llf
    LR = 2*(L2-L1)
    p = chi2.sf(LR, DF).round(3)
    return p

#%%
print(LLR_test1(model1, model2))
print(LLR_test1(model2, model3))   
print(LLR_test1(model3, model4))

#%%
print(LLR_test1(model1, model1_1))
print(LLR_test1(model2, model2_1))
print(LLR_test1(model3, model3_1))
print(LLR_test1(model4, model4_1))

#%%
print(LLR_test1(model1_1, model2_1))
print(LLR_test1(model2_1, model3_1))   
print(LLR_test1(model3_1, model4_1))

#%%
print(LLR_test1(model2_1, model2_2))

#===============================The best model is (2,0,1)======================
#%%

train = df.iloc[:-12]
test = df.iloc[-12:]

start = len(train)
end = start + len(test) - 1

#%%

model = ARIMA(train, order=(2,0,1), enforce_invertibility=False)
res = model.fit()


forecast = res.predict(start=start, end=end, dynamic=False).rename('Forecast ARIMA(2,0,1)')



ax = test.plot()
forecast.plot(ax=ax, legend=True)
#%%

from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(test,forecast))