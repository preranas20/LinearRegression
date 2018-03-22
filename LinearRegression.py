#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 13:46:56 2017
@author: preranasingh
"""
#Author:Prerana Singh
#Email:psingh17@uncc.edu


 
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import pandas as pd
import matplotlib.lines as mlines

#read data from csv file
filename = '/Users/preranasingh/Documents/sem2/ML/Quiz and  Exam-sample/Homework 1/linear_regression_test_data.csv'
data = pd.read_csv(filename,index_col=0)
data


##########Principal Component Analysis###################
#grab data for each column and calculate the mean and variance
x = data['x']
meanx = np.mean(x)
meanx
mean_difference_x = [pow((xi - meanx), 2) for xi in x]
mean_difference_x
variance_x = sum(mean_difference_x)/float(len(x) - 1)
variance_x


y = data['y']
meany = np.mean(y)
meany
mean_difference_y = [pow((yi - meany), 2) for yi in y]
mean_difference_y
variance_y = sum(mean_difference_y)/float(len(y) - 1)
variance_y


y_th= data['y_theoretical']
meany_th = np.mean(y_th)
meany_th
mean_difference_y_th = [pow((y_thi - meany_th), 2) for y_thi in y_th]
mean_difference_y_th
variance_y_th = sum(mean_difference_y_th)/float(len(y_th) - 1)
variance_y_th


#covariance between x and y
cov_x_y = np.cov(x,y)
cov_x_y


#dataframe consisting of x and y attributes
df = data.loc[:, ["x", "y"]]
df



covariance=np.cov(df.T)
covariance

#calculating eigen value and eigen vector
eigen_val ,eigen_vec = np.linalg.eig(covariance)
eigen_val
eigen_vec


eigen_pair = [(np.abs(eigen_val[i]), eigen_vec[:,i]) for i in range(len(eigen_val))]
eigen_pair


eigen_pair.sort()

eigen_pair.reverse()


total = sum(eigen_val)
var_exp = [(i / total)*100 for i in sorted(eigen_val, reverse=True)]
var_exp
cum_var_exp = np.cumsum(var_exp)
cum_var_exp


W = np.hstack((eigen_pair[0][1].reshape(2,1), eigen_pair[1][1].reshape(2,1)))
W
#PCA results
Y = df.dot(W)
Y





##########Linear Regression###################

#x=independent variable
#y=dependent variable

 #calculating valueas of beta1 and beta0  
beta1_cap=cov_x_y[0,1]/variance_x
beta1_cap
beta0_cap=meany-beta1_cap*meanx
beta0_cap

#calculating predicted y values for corresponding x
y_cap = beta0_cap + beta1_cap*x
y_cap


#Plot for PCA,inear regression,x vs y and x vs y_theoretical
fig = plt.figure()
ax=fig.add_subplot(1,1,1)
ax.plot([0,eigen_vec[0,1]],[0,eigen_vec[0,0]],color='blue')
ax.plot(x,y_cap, color = "m")
ax.scatter(x,y,color='red')
ax.scatter(x,y_th,color='green')
plt.title('PCA and linear regression on x and y')
ax.set_xlabel('PC1')
ax.set_ylabel('Predicted y')
blue_line = mlines.Line2D([], [], color='blue',markersize=15, label='PC1 loadings')
m_line = mlines.Line2D([], [], color='m',markersize=15, label='Regression line')
red_dot = mlines.Line2D([], [], color='red',marker='.',markersize=15, label='x vs y')
green_dot=mlines.Line2D([], [], color='green',marker='.',markersize=15, label='x vs y-theoretical')
plt.legend(handles=[blue_line,m_line,red_dot,green_dot])
plt.show()

#Comparing the plot for PC1 and linear regression shows that they are very similar  and are in same direction and overlaps


##############################################################################################################

#confirming the values of linear regression through built in function
from sklearn import linear_model
# do linear regression using sklearn
lm_sklearn= linear_model.LinearRegression()
x = x.reshape((len(x), 1))
lm_sklearn.fit(x, y)
y_hat = lm_sklearn.predict(x)
y_hat
