import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def cost(a, b):
    # Evaluate half MSE (Mean square error)
    m = len(Ydots)
    error = a + b * Xdots - Ydots
    J = np.sum(error ** 2) / (2 * m)
    return J

# Generate random data
Xdots = 2 * np.random.rand(100, 1)
Ydots = -5 + 7 * Xdots + np.random.randn(100, 1)

# Create a range of intervals for a and b
ainterval = np.arange(-10, 10, 0.05)
binterval = np.arange(-10, 10, 0.05)

# Create a figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Cost Function Visualization with Zoom')

# Loop through different zoom levels
for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    # Create meshgrid for the specified zoom level
    X, Y = np.meshgrid(ainterval[i * 500:(i + 1) * 500], binterval[j * 500:(j + 1) * 500])

    # Calculate cost values for the zoomed interval
    zs = np.array([cost(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    # Create a subplot
    ax = axes[i, j]
    ax = fig.add_subplot(2, 2, i * 2 + j + 1, projection='3d')  # Use add_subplot to create an Axes3D instance
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
    ax.set_xlabel('Theta0')
    ax.set_ylabel('Theta1')
    ax.set_zlabel('Cost')
    ax.set_title(f'Zoom Level {i + 1}-{j + 1}')

# Adjust layout for better spacing
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Show the plots
plt.show()
