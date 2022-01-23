# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 12:11:37 2022

@author: 91639
"""

import pandas as pd
adult_income=pd.read_csv('04+-+decisiontreeAdultincome.csv')
prep=adult_income.copy()

#print(prep.isnull().sum(axis=0))
prep=pd.get_dummies(prep,drop_first=True)

Y=prep.iloc[:,-1]
X=prep.iloc[ :,:-1]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=\
train_test_split(X,Y,test_size=0.3,random_state=1234,stratify=Y)


from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(random_state=1234)
rfc.fit(x_train,y_train)
y_predict=rfc.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_predict)
score=rfc.score(x_test,y_test)