import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
dataset = pd.read_csv('/Admission_Predict.csv')
X = dataset.iloc[:,[1]]
Y = dataset.iloc[:,-1]
X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
regressor = LinearRegression()
regressor.fit(X_train,Y_train)
y_pred= regressor.predict(X_test)
plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Linear Regression')
plt.xlabel('GRE Score')
plt.ylabel('Chance of Admit')
plt.show()
plt.scatter(X_test,Y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Linear Regression')
plt.xlabel('GRE Score')
plt.ylabel('Chance of Admit')
plt.show()
from math import sqrt
RSS = sum((Y_test-y_pred)**2)
RSE = sqrt(RSS/(len(Y_test)-2))
print("RSS IS: " ,RSS )
print("RSE IS: ",RSE)