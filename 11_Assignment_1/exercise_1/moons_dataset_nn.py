# -*- coding: utf-8 -*-
"""
Created on Mon December 9 15:16:37 2018

@author: sila
"""

from sklearn.datasets import make_moons;
import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np

from matplotlib.colors import ListedColormap, colorConverter

cm3 = ListedColormap(['#0000aa', '#ff2020', '#50ff50'])
cm2 = ListedColormap(['#0000aa', '#ff2020'])

from matplotlib import pyplot
from pandas import DataFrame

from plot_util import plot_classifier_prediction



# generate 2d classification dataset
X, y = make_moons(n_samples=100, noise=0.1)
# scatter plot, dots colored by class value
df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'red', 1:'blue'}
fig, ax = pyplot.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])

pyplot.title('Moons Dataset')
pyplot.show()

# split the dataset into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# neural network 
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(
    solver='lbfgs', 
    random_state=0, 
    hidden_layer_sizes=(10, 10), 
    activation='logistic',
)
mlp.fit(X_train, y_train)

plot_classifier_prediction(mlp, X_train, y_train, 'Neural Network')

# logistic regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

plot_classifier_prediction(logreg, X_train, y_train, 'Logistic Regression')

# kmeans
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2)
kmeans.fit(X_train)
labels = kmeans.labels_

colormap = np.array(['red', 'blue', 'yellow', 'green', 'red']) 
plt.scatter(X_train[:,0], X_train[:,1], c=colormap[labels])
plt.title('K Means Classification')
plt.show()

# decision tree
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=5, random_state=0)
tree.fit(X_train, y_train)

plot_classifier_prediction(tree, X_train, y_train, 'Decision Tree')

# random forest
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=7, random_state=0)
forest.fit(X_train, y_train)

plot_classifier_prediction(forest, X_train, y_train, 'Random Forest')

# support vector machine
from sklearn import svm

# svc = svm.LinearSVC(C=0.1)
# svc = svm.SVC(kernel='poly', degree=7, C=10)
svc = svm.SVC(kernel='rbf', gamma=2, C=1)
svc.fit(X_train, y_train)

plot_classifier_prediction(svc, X_train, y_train, 'Support Vector Machine')