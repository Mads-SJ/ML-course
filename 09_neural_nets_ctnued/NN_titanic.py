import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns

# load the data
data = pd.read_csv(r"C:\Users\Mads\Desktop\datamatiker\4. semester\ML\ML-code\09_neural_nets_ctnued\titanic_train_500_age_passengerclass.csv", header=0, sep=",");

#clean the data
data.drop(["PassengerId"], axis=1, inplace=True)

data.dropna(inplace=True)
# avg = data["Age"].mean()
# data["Age"].fillna(avg, inplace=True)

y = data["Survived"]
data.drop(["Survived"], axis=1, inplace=True)
X = data
print(X)
print(y)

# Create a figure with two subplots
fig, axs = plt.subplots(2,2)

# Create the first plot in the first subplot
sns.boxplot(x=X["Pclass"], y=X["Age"], ax=axs[0, 0])
axs[0, 0].set_title('Age vs Pclass')

df = pd.DataFrame({'Age': X["Age"], 'Survived': y})

# Create histograms for the age of survivors and non-survivors
df[df['Survived'] == 0]['Age'].hist(bins=30, alpha=0.5, ax=axs[0, 1])
df[df['Survived'] == 1]['Age'].hist(bins=30, alpha=0.5, ax=axs[0, 1])

# Create a histogram for all ages
df['Age'].hist(bins=30, alpha=0.5, ax=axs[0, 1], histtype='step', color='black')

# Set the title and labels
axs[0, 1].set_title('Age vs Survived')
axs[0, 1].set_xlabel('Age')
axs[0, 1].set_ylabel('Count')

# Add a legend
axs[0, 1].legend(['Total', 'Died', 'Survived'])


survival_counts = X.groupby(['Pclass', y]).size().unstack()

# Create the plot
survival_counts.plot(kind='bar', stacked=True, ax=axs[1, 0])

# Name the axes
axs[1, 0].set_xlabel('Pclass')
axs[1, 0].set_ylabel('Count')

# Add a legend
axs[1, 0].legend(['Died', 'Survived'])

# Show the figure with the subplots
# plt.show()

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#scale the data
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# create the model
mlp = MLPClassifier(
    hidden_layer_sizes=(100, 100, 100, 100, 100), 
    max_iter=100000,
    activation='relu',
    random_state=0
)

# train the model
mlp.fit(X_train, y_train)

# evaluate the model
predictions = mlp.predict(X_test)
matrix = confusion_matrix(y_test, predictions)
TN, FP, FN, TP = matrix.ravel()
print("Presicion: ", TP/(TP+FP))
print("Recall: ", TP/(TP+FN))
print("Accuracy: ", (TP+TN)/(TP+FP+FN+TN))
print (classification_report(y_test,predictions))

x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Predict the class for each point in the meshgrid
Z = mlp.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the predictions
plt.contourf(xx, yy, Z, alpha=0.8)
scatter = plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolor='k')
# name the axes
plt.title('MLP prediction')
plt.xlabel('Pclass')
plt.ylabel('Age')
plt.legend(handles=scatter.legend_elements()[0], labels=['Died', 'Survived'])
plt.show()