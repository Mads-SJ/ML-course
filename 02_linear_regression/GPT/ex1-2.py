import numpy as np
import matplotlib.pyplot as plt

# Generate data
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Plot the data
plt.scatter(X, y, alpha=0.5, label='Data Points')
plt.title('Scatter Plot of Generated Data')
plt.xlabel('X')
plt.ylabel('y')
#plt.legend()
plt.grid(True)
plt.show()