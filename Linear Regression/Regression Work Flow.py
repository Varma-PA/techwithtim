import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import pickle
import matplotlib.pyplot as plt
from matplotlib import style

data = pd.read_csv("../student/student-mat.csv", sep=";")

reformattedData = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

# Removing G3
X = np.array(reformattedData.drop(predict, 1))

# G3 Values from dataset
y = np.array(reformattedData[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

best = 0

# for _ in range(150):
#
#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
#
#     linear = linear_model.LinearRegression()
#
#     linear.fit(x_train, y_train)
#
#     accuracy = linear.score(x_test, y_test)
#
#     print(accuracy)
#
#     if(accuracy > best):
#         best = accuracy
#         with open("studentmodel.pickle", "wb") as f:
#             pickle.dump(linear, f)

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

#To print the accuracy
print((linear.score(x_test, y_test))*100)

print("Coefficient : \n", linear.coef_)

print("Intercept: \n", linear.intercept_)

predictions = linear.predict(x_test)

print(len(predictions))

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

p = 'G2'

style.use("ggplot")

plt.scatter(reformattedData[p], reformattedData["G3"])

plt.xlabel(p)
plt.ylabel("G3")

plt.show()
