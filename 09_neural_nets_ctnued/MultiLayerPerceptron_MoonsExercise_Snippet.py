# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 17:04:39 2021

@author: sila
"""

#This is a short example of using a MLpClassifier to predict data.
# Here using a Neural Net to predict datapoints from the Moons dataset.

#First we will import the data and just visualize them to get an idea of how it looks.

import matplotlib.pyplot as plt
from sklearn.datasets import make_moons;
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import numpy as np
# generate 2d classification dataset
X, y = make_moons(n_samples=400, noise=0.1)
# scatter plot, dots colored by class value
df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'red', 1:'blue'}
fig, ax = plt.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10, 10, 10), max_iter=1000)
mlp.fit(X_train, y_train)

h = .02  # step size in the mesh

# create a mesh to plot in
x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

plt.figure()
plt.subplot(1, 1, 1)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

Z = mlp.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

# Plot also the training points
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
plt.xlabel('Y')
plt.ylabel('X')
plt.title("MLP")

plt.show()