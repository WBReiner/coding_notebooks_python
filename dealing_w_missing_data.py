#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 16:46:30 2018

@author: whitneyreiner
"""
#listwise deletion
# In python
mydata.dropna(inplace=True)

# =============================================================================
#THIS IS R, not Python
 #Pairwise Deletion
# ncovMatrix <- cov(mydata, use="pairwise.complete.obs")
# #Listwise Deletion
# ncovMatrix <- cov(mydata, use="complete.obs")
# =============================================================================

#Dropping Vars
del mydata.column_name
mydata.drop('column_name', axis=1, inplace=True)



#IMPUTATION

In Python
from sklearn.preprocessing import Imputer
values = mydata.values
imputer = Imputer(missing_values=’NaN’, strategy=’mean’)
transformed_values = imputer.fit_transform(values)
# strategy can be changed to "median" and “most_frequent”


# KNN
from fancyimpute import KNN    

# Use 5 nearest rows which have a feature to fill in each row's missing features
knnOutput = KNN(k=5).complete(mydata)