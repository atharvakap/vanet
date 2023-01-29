import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("overtaking.csv")

# input
x = dataset.iloc[:, [0, 4]].values

# output
y = dataset.iloc[:, 5].values

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
xtrain = sc_x.fit_transform(x_train)
xtest = sc_x.transform(x_test)

print(xtrain[0:10, :])

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)
classifier.fit(xtrain, y_train)

y_pred = classifier.predict(xtest)
print("Chance of an accident: ", y_pred)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix: \n", cm)

from sklearn.metrics import accuracy_score

print("Accuracy: ", accuracy_score(y_test, y_pred))

from matplotlib.colors import ListedColormap

X_set, y_set = xtest, y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1,
                               stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1,
                               stop=X_set[:, 1].max() + 1, step=0.01))

plt.contourf(X1, X2, classifier.predict(
    np.array([X1.ravel(), X2.ravel()]).T).reshape(
    X1.shape), alpha=0.75, cmap=ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)

plt.title('Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Chance of an accident')
plt.legend()
plt.show()
