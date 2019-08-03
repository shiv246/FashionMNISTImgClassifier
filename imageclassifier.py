import tensorflow as tf
from tensorflow import keras
import numpy as np

#importing dataset
fashion_mnist = keras.datasets.fashion_mnist

#Splitting dataset into Test set and Train set
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

#Splitting training dataset into validation and training 
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
"Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

#creating the layers for the model
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy",
    optimizer="sgd",

    metrics=["accuracy"])

#Training the Model
history = model.fit(X_train, y_train, epochs=30,
     validation_data=(X_valid, y_valid))

#Evaluating model with test data
model.evaluate(X_test, y_test)

# Test - Predict on the first 3 instances
X_new = X_test[:3]
y_pred = model.predict_classes(X_new)
y_pred_classes = np.array(class_names)[y_pred]
print(y_pred_classes)

