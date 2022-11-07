# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 12:36:07 2022

@author: IBRAHIM MUSTAPHA
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

estate = pd.read_csv('D:/Practical/data/real_estate.csv')

#X = estate.iloc[:, :-1].values
#y = estate.iloc[:, 6].values 

#Understanding the data
# Shape of our dataset
estate.shape

# Info our dataset
estate.info()

# Describe our dataset
estate.describe()

print(estate.head())

#Exploratory Data Analysis
sns.set(font_scale=1.15)
plt.figure(figsize=(8,4))
sns.heatmap(
   estate.corr(),        
    cmap='RdBu_r', 
    annot=True, 
    vmin=-1, vmax=1);
sns.jointplot(x='Transaction_Date',y='House_price',data=estate)
sns.jointplot(x='House_Age',y='House_price',data=estate)
sns.pairplot(estate)

         
#y = estate['House_price']
#X = estate[['Transaction_Date', 'House_Age','Distance_to_nearest_MRT_station', 'Number_of_convenience_stores']]
y = estate.iloc[:, 6].values 
X = estate.iloc[:, :-1].values   
  
#splittting the data into Train and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

#Feature Scaling with using those estimated parameters (mean & standard deviation)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#Fitting multiple linear Regression to the Training set
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train_std, y_train)
#model.fit(X_train, y_train)


#Model Coefficients
modelcoef = model.coef_
print('Coefficients: ', modelcoef)

#Model Intercept
intr =model.intercept_
print('Intercept: ', intr)

#Calculating house price
x1, x2, x3, x4, x5, x6  = modelcoef

def house_price(var1, var2, var3, var4, var5, var6):
    pred_price = intr + var1 * x1 + var2 * x2 + var3 * x3 + var4 * x4 + var5 * x5 + var6 * x6 
    return pred_price/50.0
    
    
#X_test_std = X_test
#Predicting the Test set results
y_pred = model.predict(X_test_std)
plt.scatter(y_test, y_pred)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')

#Calculating the evaluating metrics
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Square Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#Calculating the R squared value
from sklearn.metrics import r2_score
print('R squared Error:', r2_score(y_test, y_pred))

print("Misclassified examples %d" %(y_test !=y_pred).sum())
 
from sklearn.metrics import r2_score
print("Accuracy Score %0.3f" % r2_score(y_test, y_pred))

a, b, c, d, e, f =[2013.333, 6.3, 90.45606, 9, 24.97433, 121.5431]

pred_price=house_price(a, b, c, d, e, f)

print("Predicted House Price is : " , pred_price)

pickle.dump(model, open('model.pkl','wb')) 
"""
def plot_the_loss_curve(epochs, rmse):
 # Plot the loss curve, which shows loss vs. epoch.

  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Root Mean Squared Error")

  plt.plot(epochs, rmse, label="Loss")
  plt.legend()
  plt.ylim([rmse.min()*0.97, rmse.max()])
  plt.show()
  #plot_the_loss_curve(y_pred, np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
        #from sklearn.metrics import r2_score
  
"""

"""
coeffecients = pd.DataFrame(model.coef_,X.columns)
coeffecients.columns = ['Coeffecients']
coeffecients
"""


#def mod_predict(var1, var2, var3):
#    profit=var1*coef(1) + var2*coef(2)+var3*coef(3)+ct*coef(4) -inter
 #   return profit
#x1=[]
#print(mod_predict(var1=165349.20, var2=136897.80, var3=471784.10))
#pickle.dump(model, open('model.pkl','wb')) 
#
 