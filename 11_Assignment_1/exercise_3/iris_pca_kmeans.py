from sklearn import datasets
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

iris_df = datasets.load_iris()

pca = PCA(2)

X, y = iris_df.data, iris_df.target
X_proj = pca.fit_transform(X)

colormap = np.array(['red','blue', 'yellow', 'green', 'red'])

kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X_proj)

labels = kmeans.labels_

# Plot the Original Classifications
plt.subplot(1, 2, 1)
plt.scatter(X_proj[:,0], X_proj[:,1], c=colormap[y], s=40)
plt.title('Real Classification')
 
# Plot the Models Classifications
plt.subplot(1, 2, 2)
plt.scatter(X_proj[:,0], X_proj[:,1], c=colormap[labels], s=40)
plt.title('K Mean Classification')

plt.show()