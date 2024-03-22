from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report

X, y = load_digits(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

mlp = MLPClassifier(
    hidden_layer_sizes=(10, 10, 10), 
    max_iter=1000,
    activation='relu',
    random_state=0
)

mlp.fit(X_train, y_train)

predictions = mlp.predict(X_test)
print(predictions)
matrix = confusion_matrix(y_test, predictions)
print(matrix)
print (classification_report(y_test,predictions))

# plot predictions for the first 10 images in the test set
fig, axes = plt.subplots(2, 5, figsize=(5, 3))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(8, 8), cmap='gray')
    ax.set_title(f"tar: {y_test[i]}, pre: {predictions[i]}")
    ax.axis('off')

plt.show()