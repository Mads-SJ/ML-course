from sklearn import datasets
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from scipy.stats import mode

# Function to map the predicted labels to the true labels
# This function is written by CHATGPT
def map_labels(y_true, y_pred):
    # Create an array of zeros with the same shape as the predicted labels
    new_predicted_y = np.zeros_like(y_pred)
    
    # For each unique label in the predicted labels
    for i in np.unique(y_pred):
        # Create a mask for the current label
        mask = (y_pred == i)
        
        # Assign the most common true label to the new labels
        new_predicted_y[mask] = mode(y_true[mask])[0]
    
    return new_predicted_y

iris_df = datasets.load_iris()

pca = PCA(2)

X, y = iris_df.data, iris_df.target
X_proj = pca.fit_transform(X)

colormap = np.array(['red','blue', 'yellow', 'green'])

kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X_proj)

predicted_y = map_labels(y, kmeans.labels_)

# Plot the Original Classifications
plt.subplot(1, 2, 1)
plt.scatter(X_proj[:,0], X_proj[:,1], c=colormap[y], s=40)
plt.title('Real Classification')
 
# Plot the Models Classifications
plt.subplot(1, 2, 2)
plt.scatter(X_proj[:,0], X_proj[:,1], c=colormap[predicted_y], s=40)
plt.title('K Mean Classification')

plt.show()

# calculate the accuracy of the model
print(y)
print(predicted_y)
accuracy = accuracy_score(y, predicted_y)
print("Accuracy: ", accuracy)