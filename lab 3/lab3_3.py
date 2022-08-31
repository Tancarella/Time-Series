import numpy as np
import pandas as pd
from statsmodels.tsa.api import SimpleExpSmoothing, Holt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

df=pd.read_csv('C:/Users/Admin/Desktop/UCZELNIA/2022_2023/Time Series/Laboratory/Lab3/IBM.csv', 
               index_col='Date', parse_dates=True)

print(df)

#Choosing only one column from original data
print('\nOnly open column=================')
df=df['Open']
print(df)

#ploting for finding longest linear trend

#==============================================================================
# df.plot().legend()  #It shouldn't be comment if you want to see plot - just for clarity 
#==============================================================================

# I guess it's maybe 05.02.1962 - 15.06.1962 
# or 15.06.1962 - 01.01.1964 (longer but I'm not sure if its enought straigth)
# let's take second option

#df_line = df.loc['19620205':'19620615']  #first option
df_line = df.loc['19620615':'19640101']

print('\nNew data: ==================================')
print(df_line.head())

#==============================================================================
#df_line.plot().legend() #ALSO shouldn't be comment to see plot
#==============================================================================

print(df_line.index)
print('\nNew frequency=================')

df_line=df_line.asfreq(freq='B')

print(df_line.index)

print('\n===================================')
print(df_line)

#Nans appeared due to lack of data in some days
# Let's remove them by forward fill

df_line=df_line.ffill(axis=0)

# Now let's go to Holt and SES forecasting


ncut=int(0.8*len(df_line))
print(ncut)

train_data=df_line.iloc[:ncut]
test_data=df_line.iloc[ncut:]

#ax=train_data.plot()
#test_data.plot(ax=ax)
#ax.legend(['Train', 'Test'])


fitSES=SimpleExpSmoothing(train_data).fit()
fcastSES=fitSES.forecast(len(test_data)).rename('SES predict')

fitHolt=Holt(train_data,exponential=False).fit()
fcastHolt=fitHolt.forecast(len(test_data)).rename('Holt predict')

ax=train_data.plot()
test_data.plot(ax=ax)
fcastSES.plot(ax=ax)
fcastHolt.plot(ax=ax)
ax.legend(['Train', 'Test', 'Predicted SES', 'Predictet Holt'])

print('\nMean absolute error for SES and Holt in this order:')
print(mean_absolute_error(test_data, fcastSES))
print(mean_absolute_error(test_data, fcastHolt))
