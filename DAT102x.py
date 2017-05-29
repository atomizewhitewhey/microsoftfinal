#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 00:09:23 2017

@author: matthewyeozhiwei
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import math
import time

cust = pd.read_csv('/Users/matthewyeozhiwei/repos/MicrosoftFinal/AWCustomers.csv')
sales = pd.read_csv('/Users/matthewyeozhiwei/repos/MicrosoftFinal/AWSales.csv')

## Join the Customer Sales Data and the CustomerID Data
X = pd.merge(cust, sales, how = 'inner', on = 'CustomerID')


str_cols = ['CustomerID', 'Title', 'FirstName', 'MiddleName', 'LastName', 'Suffix', 
            'AddressLine1', 'AddressLine2', 'City', 'StateProvinceName', 
            'CountryRegionName', 'PostalCode', 'PhoneNumber',
            'LastUpdated']
gender = ['M', 'F']
order_categories = ['Clerical', 'Skilled Manual','Manual','Management','Professional']
marital = ['M', 'S']
years = ['1998-01-01', '1992-01-01' ,'1987-01-01', '1967-01-01']


def clean(data):
    data = data.drop_duplicates(subset=['AvgMonthSpend', 'CustomerID'])
    X1 = data.drop(labels = str_cols, axis = 1)
    X1['BirthDate'] = pd.to_datetime(X1['BirthDate'])
    X1['Year'] = X1['BirthDate'].dt.year
    X1['Age'] = X1['Year'].apply(lambda x: 2017 - x)
    X1 = X1.drop(labels = ['BirthDate', 'Year'], axis = 1)
    return X1

X1 = clean(X)

## Function to plot histograms
def plot_hist(col):
    fig = plt.figure(figsize = (9,9))
    ax = fig.gca()
    plt.title('Histogram of ' + col)
    plt.ylabel('Count')
    plt.xlabel(col)
    plt.hist(X1[col])

## Function to plot conditional histograms
def plot_condhist(col, hist_col):
    grid1 = sns.FacetGrid(X1, col)
    grid1.map(plt.bar, hist_col.value_counts() , alpha = .7)

## Function to plot conditional 5d scatter plots
def plot_condscatter5d(col, row, hue, x, y):
    g = sns.FacetGrid(X1, col= col, row = row, 
                      hue = hue, palette="Set2", margin_titles=True)
    g.map(sns.regplot, x, y, fit_reg = False)

## Function to plot conditional 3d scatter plots
def plot_condscatter3d(col, x, y):
    g = sns.FacetGrid(X1, col= col,  
                      palette="Set2", margin_titles=True)
    g.map(sns.regplot, x, y, fit_reg = False)    


        
    
## Function to plot scatter plot matrix

num_cols = ['AvgMonthSpend', 'YearlyIncome', 'TotalChildren', 'NumberChildrenAtHome',
            'NumberCarsOwned', 'Age']
sns.pairplot(X1[num_cols], size=2)


## Function to plot boxplots
def auto_boxplot(df, plot_cols, column):
    for col in plot_cols:
        fig = plt.figure(figsize=(9, 6))
        ax = fig.gca()
        df.boxplot(column = column, by = col, ax = ax)
        ax.set_title('Box plots of Log ' + column + ' by ' + col)
        ax.set_ylabel('Log ' + column)
    return column 

plot_cols = ['Occupation', 'Education', 'NumberChildrenAtHome', 'TotalChildren', 'NumberCarsOwned']



fig = plt.figure()
ax = plt.subplot(1,2,1)
ax.set_title('Bike Buyer = 0')
X1[X1['BikeBuyer'] == 0]['Occupation'].value_counts().plot(kind = 'bar')
ax = plt.subplot(1,2,2)
ax.set_title('Bike Buyer = 1')
X1[X1['BikeBuyer'] == 1]['Occupation'].value_counts().plot(kind = 'bar')

fig = plt.figure()
ax = fig.gca()
X1.MaritalStatus.value_counts().plot(kind = 'bar')

                      
## Edit MetaData: Map Categorical Features to Numerals
def metadataedit(X1):
    X1.Gender = X1.Gender.map({'M':0, 'F':1})
    X1.MaritalStatus = X1.MaritalStatus.map({'M' : 0, 'S' : 1})
    X1 = pd.get_dummies(X1, columns = ['Education', 'Occupation'])
    return X1

X1 = metadataedit(X1)

## Final check for any null values/ Ensure that all features are numerical

print(X1[pd.isnull(X1).any(axis=1)])
print(X1.dtypes)
print(X1.head())

## Get ready data for model fitting (Xr is for regression, Xc is for classification)
yc = X1['BikeBuyer']
yr = X1['AvgMonthSpend']
Xc = X1.drop(labels = ['BikeBuyer','AvgMonthSpend'], axis = 1)
Xr = Xc

## Split Data
from sklearn.model_selection import train_test_split
Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size = 0.4, random_state = 7)
Xc_test = pd.DataFrame(Xc_test)

Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size = 0.3, random_state = 7)
Xr_test = pd.DataFrame(Xr_test)

## Normalize the Data using Standardisation

from sklearn import preprocessing
stand = preprocessing.StandardScaler()
stand.fit(Xc_train)
Xc_train = stand.transform(Xc_train)
Xc_test = stand.transform(Xc_test)

stand = preprocessing.StandardScaler()
stand.fit(Xr_train)
Xr_train = stand.transform(Xr_train)
Xr_test = stand.transform(Xr_test)


## Classification Model - Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier
best_score = 0
fcmodel = RandomForestClassifier(n_estimators = 30, max_depth = 10, oob_score = True, random_state = 0)
fcmodel.fit(Xc_train, yc_train)
score = fcmodel.score(Xc_test, yc_test)
print("Classification Score: ", round(score*100, 3))

from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators' : [30,35,40],
                 'max_depth' : [10,15,20]}
optfcmodel = GridSearchCV(fcmodel, parameters)
optfcmodel.fit(Xc_train, yc_train)
score = optfcmodel.score(Xc_test, yc_test)
print("Optimized Classification Score: ", round(score*100, 3))

imptlist = fcmodel.feature_importances_.tolist()
import heapq
print(imptlist[8] + imptlist[9] + imptlist[10] + imptlist[11] + imptlist[12])
print(imptlist[13] + imptlist[14] + imptlist[15] + imptlist[16] + imptlist[17])
temp = []
for i in imptlist:
    temp.append([i])
temp = pd.DataFrame(temp, index = ['Gender', 'MaritalStatus', 'HomeOwnerFlag', 'NumberCarsOwned',
       'NumberChildrenAtHome', 'TotalChildren', 'YearlyIncome', 'Age',
       'Education_Bachelors', 'Education_Graduate Degree',
       'Education_High School', 'Education_Partial College',
       'Education_Partial High School', 'Occupation_Clerical',
       'Occupation_Management', 'Occupation_Manual', 'Occupation_Professional',
       'Occupation_Skilled Manual'])
temp.columns = ['Feature Importance']
print(temp)

import sklearn.metrics as metrics

yc_predict = fcmodel.predict(Xc_test)
print(metrics.confusion_matrix(yc_test, yc_predict))
print('Recall Score:', round(metrics.recall_score(yc_test, yc_predict) * 100,3))
print('Accuracy Score:', round(metrics.accuracy_score(yc_test, yc_predict) * 100, 3))
print('Precision:', round(metrics.precision_score(yc_test, yc_predict) * 100, 3))
print('F1 Score:', round(metrics.f1_score(yc_test, yc_predict) * 100,3))
fpr, tpr, threshold = metrics.roc_curve(yc_test, yc_predict)
roc_auc = metrics.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc, color='darkorange')
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--', color = 'b')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


## Regression Model - GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor()
gbr.fit(Xr_train, yr_train)
print("Regression Score: ", round(gbr.score(Xr_test, yr_test) * 100,3))

import scipy.stats
from sklearn.model_selection import GridSearchCV
parameter_dist = {'loss' : ['ls', 'lad', 'huber', 'quantile'],
                  'learning_rate' : [0.1,0.15,0.20],
                  'n_estimators' : [30,35,40]
                    }
optgbr = GridSearchCV(gbr, parameter_dist)
optgbr.fit(Xr_train, yr_train)
score = optgbr.score(Xr_test, yr_test)
print("Optimized Regression Score: ", round(score * 100,3))

yr_predict = gbr.predict(Xr_test)
from math import sqrt
print('Root Mean Squared Error:',sqrt(metrics.mean_squared_error(yr_test, yr_predict)))
plt.scatter(yr_predict, yr_test)
plt.ylabel('Log of AvgMonthSpend')
plt.xlabel('Scored Labels')



