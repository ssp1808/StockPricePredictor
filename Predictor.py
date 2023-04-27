import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

address = "E:\Learning\Python\StockPricePredictor\HistData.csv"

df = pd.read_csv(address)
df.columns=['Sr','Date','Open','High','Low','Close','AdjClose','Volume']
df  = df.dropna()
# df  =  df[df['id'].apply(lambda x: is_float(x))]

df = df[~(df['Volume'] == 0)] # remove outliers
df = df.drop('Date', axis=1) 
df = df[df["Open"].str.contains("Dividend")==False]

df = df.iloc[:98,:]
print(df.head)
model = LinearRegression() 

#Split the data
X = df.drop('Close', axis=1)
y = df['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Fitting the model

model.fit(X_train,y_train)
result = model.predict(X_test)

#Check result
print('Mean Squared Error = ', mean_squared_error(y_test,result))
