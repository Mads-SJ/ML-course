# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 13:41:07 2021

@author: sila
"""

#import panda library and a few others we will need.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

col_names = ["Id", "Churn", "Line", "Grade", "Age", "Distance", "StudyGroup"]

data = pd.read_csv(r'C:\Users\Mads\Desktop\datamatiker\4. semester\ML\ML-code\12_convolution\StudentChurn.csv', header=0, sep=";")

# show the data
print ( data .describe( include = 'all' ))
#the describe is a great way to get an overview of the data
print ( data .values)

print(data.columns)
print(data.shape)

data.drop('Id', axis=1, inplace=True)

data.replace(['HTX', 'STX', 'HHX', 'EUX', 'HF'], [0, 1, 2, 3, 4], inplace=True)
data.replace(['Stopped', 'Completed'], [0, 1], inplace=True)

data.dropna(inplace=True)
y = data["Churn"]
X = data.drop(["Churn"], axis=1)
print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


lrc = LogisticRegression(random_state=0, C=0.5)
lrc.fit(X_train, y_train)

print("LOGISTIC REGRESSION CLASSIFIER")
predictions = lrc.predict(X_test)
print(classification_report(y_test,predictions))

rfc = RandomForestClassifier(random_state=42)

rfc.fit(X_train, y_train)

print("RANDOM FOREST CLASSIFIER")
predictions = rfc.predict(X_test)
print(classification_report(y_test,predictions))