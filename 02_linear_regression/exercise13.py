# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 20:54:51 2018

@author: Sila
"""

# Python version
import numpy as np
import matplotlib.pyplot as plt


def cost(a,b,X,y):
     ### Evaluate half MSE (Mean square error)
     m = len(y)
     error = a*X + b - y
     J = np.sum(error ** 2)/(2*m)
     return J

X = 2 * np.random.rand(100, 1)
print(X)
y = 4 + 3 * X + np.random.randn(100, 1)

ainterval = np.arange(0.5,5, 0.01)
binterval = np.arange(1,10, 0.01)

low = cost(0,0, X, y)
bestatheta = 0
bestbtheta = 0
for atheta in ainterval:
    for btheta in binterval:
        # print("xy: %f:%f:%f" % (atheta,btheta,cost(atheta,btheta, X, y)))
        if (cost(atheta,btheta, X, y) < low):
           low = cost(atheta,btheta, X, y)
           bestatheta = atheta
           bestbtheta = btheta

print("a and b: %f:%f" % (bestatheta, bestbtheta))

# Plot the data and the linear regression line
plt.scatter(X, y, alpha=0.5, label='Data Points')
plt.plot(X, bestatheta + bestbtheta*X, color='red', label='Linear Regression Line')
plt.title('Plot of Generated Data and Linear Regression Line')
plt.grid(True)
plt.show()