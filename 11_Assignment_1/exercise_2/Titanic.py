import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.impute import SimpleImputer

# CHANGE THIS PATH TO YOUR OWN PATH
data = pd.read_csv(r"C:\Users\Mads\Desktop\datamatiker\4. semester\ML\ML-code\11_Assignment_1\exercise_2\titanic_800.csv", header=0, sep=",")

#remove redundant columns
data.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)

print(data.shape)
data.dropna(inplace=True)
print(data.shape)
# avg = data["Age"].mean()
# data["Age"].fillna(avg, inplace=True)

data.replace(["male", "female"], [0, 1], inplace=True)
data.replace(["S", "C", "Q"], [0, 1, 2], inplace=True)

X = data.drop(["Survived"], axis=1)
y = data["Survived"]

fig, axs = plt.subplots(2,2)

sns.boxplot(x=X["Pclass"], y=X["Age"], ax=axs[0, 0])
axs[0, 0].set_title('Age vs Pclass')

df = pd.DataFrame({'Age': X["Age"], 'Survived': y})

# Create histograms for the age of survivors and non-survivors
df[df['Survived'] == 0]['Age'].hist(bins=30, alpha=0.5, ax=axs[0, 1])
df[df['Survived'] == 1]['Age'].hist(bins=30, alpha=0.5, ax=axs[0, 1])

df['Age'].hist(bins=30, alpha=0.5, ax=axs[0, 1], histtype='step', color='black')

axs[0, 1].set_title('Age vs Survived')
axs[0, 1].set_xlabel('Age')
axs[0, 1].set_ylabel('Count')

axs[0, 1].legend(['Total', 'Died', 'Survived'])


survival_counts = X.groupby(['Pclass', y]).size().unstack()

survival_counts.plot(kind='bar', stacked=True, ax=axs[1, 0])

axs[1, 0].set_xlabel('Pclass')
axs[1, 0].set_ylabel('Count')

axs[1, 0].legend(['Died', 'Survived'])

# plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

imp = SimpleImputer(strategy='mean')
imp_train = imp.fit(X_train)
X_train = imp_train.transform(X_train)
X_test = imp_train.transform(X_test)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

rfc = RandomForestClassifier(n_estimators=10, random_state=0)

rfc.fit(X_train, y_train)

print("RANDOM FOREST CLASSIFIER")
predictions = rfc.predict(X_test)
matrix = confusion_matrix(y_test, predictions)
TN, FP, FN, TP = matrix.ravel()
print("Presicion: ", TP/(TP+FP))
print("Recall: ", TP/(TP+FN))
print("Accuracy: ", (TP+TN)/(TP+FP+FN+TN))
print(classification_report(y_test,predictions))

mlp = MLPClassifier(
    hidden_layer_sizes=(10, 10, 10), 
    max_iter=1000,
    activation='relu',
    random_state=0
)

mlp.fit(X_train, y_train)

print("MLP CLASSIFIER")
predictions = mlp.predict(X_test)
matrix = confusion_matrix(y_test, predictions)
TN, FP, FN, TP = matrix.ravel()
print("Presicion: ", TP/(TP+FP))
print("Recall: ", TP/(TP+FN))
print("Accuracy: ", (TP+TN)/(TP+FP+FN+TN))
print(classification_report(y_test,predictions))
