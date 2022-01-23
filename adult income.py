# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 23:45:12 2022

@author: 91639
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

adult_income=pd.read_csv('04+-+decisiontreeAdultincome.csv')

income_prep=adult_income.copy()
#print(income_prep.isnull().sum(axis=0))
income_prep=pd.get_dummies(income_prep,drop_first=True)

Y=income_prep.iloc[:,-1]
X=income_prep.iloc[:,:-1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=1234,stratify=Y)


from sklearn.tree import DecisionTreeClassifier

#Train the Model

dtc=DecisionTreeClassifier(random_state=1234)
dtc.fit(x_train,y_train)
y_predict=dtc.predict(x_test)

#evaluate the model

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_predict)
score=dtc.score(x_test,y_test)