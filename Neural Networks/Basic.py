import tensorflow as tf
# from tensorflow import keras
import keras
import numpy as np
import matplotlib.pyplot as plt
import pickle

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# print(class_names[train_labels[6]])

train_images = train_images/255.0
test_images = test_images/255.0


# print(train_images[5])
#
# plt.imshow(train_images[1], cmap=plt.cm.binary)
#
# plt.show()
#

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    # keras.layers.Dense(64, activation="relu"),
    # keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=10)

# test_loss, test_accuracy = model.evaluate(test_images, test_labels)
#
# print("Tested Accuracy: ", test_accuracy)


# Saving the Model locally
# with open("dressmodel.pickle", "wb") as f:
#     pickle.dump(model, f)

#Loading the model which is saved locally
# pickle_in = open("dressmodel.pickle", "rb")
# model = pickle.load(pickle_in)


prediction = model.predict(test_images)


# print(prediction.shape)

for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediction: "+ class_names[np.argmax(prediction[i])])
    plt.show()
