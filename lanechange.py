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

xp = dataset['18'].tolist()
xpoints = np.array(xp[20:])
ypoints = np.array(y_pred)
print(xpoints)
plt.scatter(xpoints, ypoints)

plt.title('Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Chance of an accident')
plt.show()
