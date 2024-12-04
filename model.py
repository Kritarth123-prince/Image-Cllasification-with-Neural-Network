import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Load Fashion MNIST dataset
fashion = keras.datasets.fashion_mnist
(xtrain, ytrain), (xtest, ytest) = fashion.load_data()

# Display a sample image and its label
imgIndex = 9
image = xtrain[imgIndex]
print("Image Label :", ytrain[imgIndex])
plt.imshow(image)

# Check the shape of the training and test data
print(xtrain.shape)
print(xtest.shape)

# Building a Neural Network Architecture
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

# Display model summary
print(model.summary())

# Prepare validation and training datasets
xvalid, xtrain = xtrain[:5000]/255.0, xtrain[5000:]/255.0
yvalid, ytrain = ytrain[:5000], ytrain[5000:]

# Compile the model
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

# Train the model
history = model.fit(xtrain, ytrain, epochs=30, validation_data=(xvalid, yvalid))

# Make predictions on the first 5 test images
new = xtest[:5]
predictions = model.predict(new)
print(predictions)

# Output the predicted classes
classes = np.argmax(predictions, axis=1)
print(classes)

# Save the model weights to a file
weights_path = 'model.h5'
model.save_weights(weights_path)
print(f"Model weights saved to {weights_path}")

