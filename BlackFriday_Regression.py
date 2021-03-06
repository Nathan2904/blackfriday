# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 19:18:13 2020

@author: GIM
"""


#loading all the required libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sn
import matplotlib.pyplot as plt
from scipy.stats import norm


data = pd.read_csv('C:\\Users\\GIM\\Desktop\\pyth\\train.csv')
data.head()
data.info()


#Unnecessary columns 
data = data.drop(['Product_ID'], axis= 1)

#data cleaning starts
missing_values = data.isnull().sum().sort_values(ascending = False)
data=data.fillna(0)


gender = np.unique(data['Gender'])
gender
age = np.unique(data['Age'])
age

#Masking categorical data
data['Gender1']=data.Gender.map({'F':0,'M':1})
data['Age1']=data.Age.map({'0-17':0,'18-25':1,'26-35':2,'36-45':3,'46-50':4,'51-55':5,'55+':6})

city_category = np.unique(data['City_Category'])
city_category
data['City_Category1']=data.City_Category.map({'A':0,'B':1,'C':2})


#changing incorrect Stay_In_Current_City_Years value of 4+ to 5
city_stay = np.unique(data['Stay_In_Current_City_Years'])
city_stay
data.loc[data.Stay_In_Current_City_Years=='4+','Stay_In_Current_City_Years']=5

#Exploratory data analysis
data[['Age','Purchase']].groupby('Age')
sn.barplot('Age', 'Purchase', data = data)
plt.show()

data[['Gender','Purchase']].groupby('Gender')
sn.barplot('Gender', 'Purchase', data = data)
plt.show()

data[['City_Category','Purchase']].groupby('City_Category')
sn.barplot('City_Category', 'Purchase', data = data)
plt.show()

#creating dummy variable
data=pd.get_dummies(data, columns=['Occupation'])
data.columns

#boxplot of Purchase
plt.boxplot(data['Purchase'])
plt.show()

#function written to remove outliers based on interquartile range
def remove_outlier(data, Purchase):
    q1 = data[Purchase].quantile(0.25)
    q3 = data[Purchase].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = data.loc[(data[Purchase] > fence_low) & (data[Purchase] < fence_high)]
    return df_out


#calling the function
data_new = remove_outlier(data,'Purchase')
print(data_new)

#boxplot after removing outliers
plt.boxplot(data_new['Purchase'])
plt.show()


#checking which occupations have a high correlation with Purchase
required=['Occupation_0', 'Occupation_1',
       'Occupation_2', 'Occupation_3', 'Occupation_4', 'Occupation_5',
       'Occupation_6', 'Occupation_7', 'Occupation_8', 
       'Occupation_12', 
       'Occupation_14', 'Occupation_15',  'Occupation_17',
       'Occupation_18', 'Purchase']

#converting to dataframe
data2 = pd.DataFrame(data_new,columns = required)


corrmat = data2.corr()
print(corrmat)


#considering only the required parameters
data_new1  = data_new[['Occupation_4','Occupation_5','Occupation_6', 'Occupation_7','Occupation_8', 'Occupation_12', 
       'Occupation_14', 'Occupation_15', 'Occupation_17','Occupation_18',
     'Purchase','Product_Category_1'
       ,'Product_Category_2','Product_Category_3','Age1','Gender1','City_Category1','Marital_Status']]

data_new1.info()



import statsmodels.api as sm

#Split into train and test dataset
X = sm.add_constant( data_new1 )
Y = data_new1['Purchase']

from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split( X ,Y,train_size = 0.8,random_state = 42 )


# Fitting the Linear Regression Model
model_1 = sm.OLS(train_y, train_X.astype(float)).fit()
model_1.summary2()
#new
var=data_new[['Occupation_7','Occupation_8', 'Occupation_12',
       'Occupation_14', 'Occupation_15', 'Occupation_16', 'Occupation_17',
     'Purchase','Product_Category_1'
       ,'Product_Category_2','Product_Category_3','Age1','Gender1','City_Category1','Marital_Status']]
X = sm.add_constant( var )
Y = var['Purchase']
model_2 = sm.OLS(train_y, train_X).fit()
model_2.summary2()
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split( X ,Y,train_size = 0.8,random_state = 42 )


from statsmodels.stats.outliers_influence import variance_inflation_factor

# Method to calculate Variance inflation factor
def get_vif_factors( X ):
    X_matrix = X.as_matrix()
    vif = [ variance_inflation_factor( X_matrix, i ) for i in range( X_matrix.shape[1] ) ]
    vif_factors = pd.DataFrame()
    vif_factors['column'] = X.columns
    vif_factors['vif'] = vif
    return vif_factors

vif_factors = get_vif_factors(X [X.columns] )
vif_factors

# Select the features that have VIF value more than 4
columns_with_large_vif = vif_factors[vif_factors.vif > 4].column

# Plot the heatmap for features with moore than 4
plt.figure( figsize = (12,10) )
sn.heatmap( X[columns_with_large_vif].corr(), annot = True );
plt.title( "Figure 4.5 - Heatmap depicting correlation between features");


#Residual Analysis

# P-P Plot
def draw_pp_plot(model_1 , title ):
    probplot = sm.ProbPlot( model_1 .resid );
    plt.figure( figsize = (8, 6) );
    probplot.ppplot( line='45' );
    plt.title( title );
    plt.show();
   
draw_pp_plot( model_1 ,"Figure 4.6 - Normal P-P Plot of Regression Standardized Residuals");

