import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
dataset = pd.read_csv('housing price.csv') 
X = dataset.iloc[:, :1] 
Y = dataset.iloc[:, 1:2] 
from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2) 
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression() 
regressor.fit(X_train, Y_train) 
Y_pred = regressor.predict(X_test) 
plt.scatter(X_train, Y_train) 
plt.plot(X_train, regressor.predict(X_train)) 
plt.title('Training set') 
plt.xlabel('ID ') 
plt.ylabel('SALE PRICE') 
plt.show() 
plt.scatter(X_test, Y_test) 
plt.plot(X_train, regressor.predict(X_train)) 
plt.title('Test set') 
plt.xlabel('ID ') 
plt.ylabel('SALE PRICE') 
plt.show() 
print('The sale price of ID 2950 is:') 
print(regressor.predict([[3000]])) 
print('The sale price of ID 3500 is:') 
print(regressor.predict([[3900]])) 