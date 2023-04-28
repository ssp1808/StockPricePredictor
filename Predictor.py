import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

address = "E:\Learning\Python\StockPricePredictor\HistData.csv"

df = pd.read_csv(address)
df.columns=['Sr','Date','Open','High','Low','Close','AdjClose','Volume']
df  = df.dropna()

#Pre-Process the data
df = df[~(df['Volume'] == 0)] # remove outliers
df = df.drop('Date', axis=1) 
df = df[~df["Open"].str.contains("Dividend")] # remove rows containing "Dividend" in the "Open" column

df = df.iloc[:98,:]
model = LinearRegression() 

#Split the data
X = df.drop('Close', axis=1)
y = df['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

#Fitting the model
model.fit(X_train,y_train)
result = model.predict(X_test)

mse = mean_squared_error(y_test, result)
print('Mean Squared Error:', mse)
RSq = r2_score(y_test, result)
print('R2 score:', RSq)

#Check result
y_test = pd.DataFrame((y_test)).reset_index(drop=True)
y_test = y_test.astype(float)

result = pd.DataFrame(result)
result.columns=['Close']
y_test.columns=['Close']

print(y_test)
print(result)
y_test.sort_index(ascending=True)
plt.plot(y_test['Close'],label='Actual')
plt.plot(result['Close'],label='Predicted')


plt.title('AAPL Stock Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

