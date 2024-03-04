import matplotlib.pyplot as plt
import random

# Generate 20 random points
num_points = 20
x_coordinates = [random.uniform(0, 10) for _ in range(num_points)]
y_coordinates = [random.uniform(0, 10) for _ in range(num_points)]

# Plot the points
plt.scatter(x_coordinates, y_coordinates, color='blue')
plt.title('Scatter Plot of 20 Random Points')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True)
plt.show()