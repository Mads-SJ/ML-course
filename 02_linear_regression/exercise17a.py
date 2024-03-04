# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 20:23:27 2018

@author: Sila
"""


import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


X = 6 * np.random.rand(100, 1) - 3
y = 0.5 * X ** 3 + 0.5 * X ** 2 + X + 2 + np.random.randn(100, 1)

plt.plot(X, y, "g.")
plt.axis([-3, 3, -20, 30])

poly_features = PolynomialFeatures(3, include_bias=False)
X_poly = poly_features.fit_transform(X)

lm = LinearRegression()
lm.fit(X_poly, y)


#fit function
f = lambda x: lm.coef_[0][2] * x ** 3 + lm.coef_[0][1] * x ** 2 + lm.coef_[0][0]*x + lm.intercept_
plt.plot(X,f(X), "b.")

plt.show()

