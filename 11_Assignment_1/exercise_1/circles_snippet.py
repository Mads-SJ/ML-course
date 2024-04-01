# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 11:23:02 2018

@author: sila
"""

from sklearn.datasets import make_circles;
import matplotlib.pyplot as plt

from matplotlib import pyplot
from pandas import DataFrame
# generate 2d classification dataset
# generate 2d classification dataset
X, y = make_circles(n_samples=100, noise=0.05)
# scatter plot, dots colored by class value
df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'red', 1:'blue'}
fig, ax = pyplot.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
pyplot.show()

# split the dataset into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# support vector machine
from sklearn import svm

svc_lin = svm.LinearSVC(C=0.1)
svc_poly = svm.SVC(kernel='poly', degree=2, C=10)
svc = svm.SVC(kernel='rbf', gamma=1, C=10)

svc_lin.fit(X_train, y_train)
svc_poly.fit(X_train, y_train)
svc.fit(X_train, y_train)

from plot_util import plot_classifier_prediction

plot_classifier_prediction(svc_lin, X_train, y_train, 'Linear SVM')
plot_classifier_prediction(svc_poly, X_train, y_train, 'Polynomial SVM')
plot_classifier_prediction(svc, X_train, y_train, 'RBF SVM')