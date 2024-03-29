# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 19:12:56 2018

@author: sila
"""

from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split



X, y = make_moons(n_samples=1000, noise=0.1)
# split the dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# check how many samples we have
print("Number of samples: " +str(len(y)))
#visulize the dataset
plt.figure()
#define colors - red, green, blue
colormap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
# pplot labxel
plt.xlabel("X")
plt.ylabel("Y")
# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=colormap,edgecolor='black', s=20)

from datetime import datetime
k = str(datetime.now())
print(k) 
#time the execution time of this cell.

tree_clf = DecisionTreeClassifier(max_depth=2) ## indicate we do not want the tree to be deeper than 2 levels
tree_clf.fit(X,y) # training the classifier
print("seed : "+ str(tree_clf.random_state))

#prediction
print("probability of point = (2,1) = "+str(tree_clf.predict_proba([[2,1]])))
print("probability of point = (4,1) = "+str(tree_clf.predict_proba([[4,1]])))
print("probability of point = (5,2) =  "+str(tree_clf.predict_proba([[5,2]])))

#for name,score in zip(iris.feature_names[2:4],tree_clf.feature_importances_):
#    print("feature importance: ",name, score)

import pydotplus 
import matplotlib.image as mpimg
import io

dot_data = io.StringIO()

export_graphviz(tree_clf,
                out_file=dot_data, # or put a filename here filename like "graph.dot", you then need to convert it into pgn
                feature_names=["X","Y"],
                class_names=["0","1"],
                rounded=True,
                filled=True)

filename = "tree.png"
pydotplus.graph_from_dot_data(dot_data.getvalue()).write_png(filename) # write the dot data to a pgn file
img=mpimg.imread(filename) # read this pgn file

plt.figure(figsize=(8,8)) # setting the size to 10 x 10 inches of the figure.
imgplot = plt.imshow(img) # plot the image.


plt.figure()
# pplot labxel
plt.xlabel("X")
plt.ylabel("Y")
# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=colormap,edgecolor='black', s=20)
#drawing the splits
plt.axhline(y=1.75, color='red', linestyle='-')
plt.axvline(x=2.45, color = 'blue',linestyle='-')

plt.show()


