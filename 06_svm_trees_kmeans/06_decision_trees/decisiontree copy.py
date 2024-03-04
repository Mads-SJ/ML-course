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

import pydotplus 
import matplotlib.image as mpimg
import io



X, y = make_moons(n_samples=1000, noise=0.1)

# plot the dataset
plt.figure()
colormap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
plt.xlabel("X")
plt.ylabel("Y")
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=colormap,edgecolor='black', s=20)

# split the dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train the classifier and check the accuracy
tree = DecisionTreeClassifier(max_depth=2)
tree.fit(X_train, y_train)
acc= tree.score( X_test, y_test)
print("accuracy = "+str(acc))

# visualize the decision tree
dot_data = io.StringIO()
export_graphviz(tree,
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
plt.show()

