import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("../Car Data Set/car.data");

print(data.head(10))

label_encoder = preprocessing.LabelEncoder()

buying = label_encoder.fit_transform(data["buying"])
maintenance = label_encoder.fit_transform(list(data["maint"]))
door= label_encoder.fit_transform(list(data["door"]))
lug_boot = label_encoder.fit_transform(list(data["lug_boot"]))
safety = label_encoder.fit_transform(list(data["safety"]))
persons = label_encoder.fit_transform(list(data["persons"]))
classType = label_encoder.fit_transform(list(data["class"]))


print(buying)

X = list(zip(buying, maintenance, door, lug_boot, safety, persons))

y = list(classType)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

model = KNeighborsClassifier(n_neighbors=9)

model.fit(x_train, y_train)

accuracy = model.score(x_test, y_test)

print(accuracy)

predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])

