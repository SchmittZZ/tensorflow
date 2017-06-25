import pandas as pd
import quandl, math
import numpy as np
## numpy is a computing library that allows to use arrays
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

##Above are featuers

##Features and label

forecast_col = 'Adj. Close'
df. fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01*len(df)))
#math.ceil can take anything to the ceiling
#rounds everything up to the nearest whole

df['label'] = df[forecast_col].shift(-forecast_out)

df.dropna(inplace=True)

##print(df.head())


## Training and testing
x = np.array(df.drop(['label'],1))
y = np.array(df['label'])
x = preprocessing.scale(x)  ##

x = x[:-forecast_out+1]
df.dropna(inplace=True)
y = np.array(df['label'])
print(len(x),len(y))
