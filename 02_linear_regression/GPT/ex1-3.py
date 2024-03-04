import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate data
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Perform linear regression
model = LinearRegression()
model.fit(X, y)
X_test = np.linspace(0, 2, 100).reshape(-1, 1)
y_pred = model.predict(X_test)

# Plot the data points
plt.scatter(X, y, alpha=0.5, label='Data Points')

# Plot the linear regression line
plt.plot(X_test, y_pred, color='red', label='Linear Regression Line')

plt.title('Linear Regression on Generated Data')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
