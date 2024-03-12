import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# load the data
data = pd.read_csv(r"C:\Users\Mads\Desktop\datamatiker\4. semester\ML\ML-code\09_neural_nets_ctnued\titanic_train_500_age_passengerclass.csv", header=0, sep=",");

#clean the data
data.drop(["PassengerId"], axis=1, inplace=True)


#TODO: change this later to delete the rows with missing data
avg = data["Age"].mean()
data.dropna(inplace=True)
#data["Age"].fillna(avg, inplace=True)

y = data["Survived"]
data.drop(["Survived"], axis=1, inplace=True)
X = data
print(X)
print(y)

# Create a figure with two subplots
fig, axs = plt.subplots(2,2)

# Create the first plot in the first subplot
axs[0, 0].scatter(X["Age"], X["Pclass"], color = 'black' , s = 20 )
axs[0, 0].set_title('Age vs Pclass')

# Create the second plot in the second subplot
# Replace "column2" and "column3" with the names of the columns you want to plot
axs[0, 1].scatter(X["Age"], y, color = 'red' , s = 20 )
axs[0, 1].set_title('Age vs Survived')

axs[1, 0].scatter(X["Pclass"], y, color = 'blue' , s = 20 )
axs[1, 0].set_title('Pclass vs Survived')

# Show the figure with the subplots
#plt.show()

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#TODO: try to scale the data
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