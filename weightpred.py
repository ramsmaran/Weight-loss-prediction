#importing required libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import linear_model, metrics

#reading the required dataset
wtloss=pd.read_csv('wtloss.csv')
wtloss

#describing the dataset
wtloss.describe()

#info of the dataset to know if there are any null values and get the count of elements
wtloss.info()

wtloss['Exercise']=wtloss['Exercise'].replace(0,10) #replacing the values of 0 with default value 10 in exercises

wtloss['Exercise'][wtloss['Exercise'] < 0] =np.nan #making the neg values to nan values

wtloss['Exercise']=wtloss['Exercise'].fillna(wtloss['Exercise'].mean()) #replacing the neg values with mean of the coloumn

wtloss['Weight'].isnull() #nan values in weight coloumn are dropped

wtloss=wtloss.dropna() #drops the rows with null values
wtloss['Exercise']=wtloss['Exercise'].astype('int64')

wtloss

cor=wtloss.corr() #corelation of all the variable for given data set
cor

sns.heatmap(cor) #heatmap of corelation between all the variables

#linear regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
x=wtloss['Exercise']
y=wtloss['Weight']
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=.2)
model=linear_model.LinearRegression()
train_x=np.array(train_x).reshape(-1,1) #reshaping the 1d arrays to 2d array
train_y=np.array(train_y).reshape(-1,1)#reshaping 1d to 2d


model.fit(train_x,train_y)


pred_y=model.predict(train_x) #model predicted values
pred_y

#multivariate regression
mul_x=wtloss[['Exercise','Days']]
y=wtloss['Weight']
mul_y=np.array(y).reshape(-1,1)
mul_model=linear_model.LinearRegression()
mul_x,mul_test_x,mul_y,mul_test_y=train_test_split(mul_x,mul_y,train_size=.8)

mul_model.fit(mul_x,mul_y)

predmul_y=mul_model.predict(mul_x) #multivariate model predicted values
predmul_y

model.score(train_x,train_y) #linear model score 

mul_model.score(mul_x,mul_y)# multivariate model score

mean_squared_error(mul_y,predmul_y) #multivariate model mean squared error

mean_squared_error(train_y,pred_y) #univariant model mean squared error

#the multivariate model is bettr than univariate model since it has greater score and lesser mean squared error
