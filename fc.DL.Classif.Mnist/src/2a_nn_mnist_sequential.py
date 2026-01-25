"""
Example of a simple neural network for MNIST classification  
using the models.Sequential API in tensorflow 2.1
Author: Sebastian Gottwald
Project: https://github.com/sgttwld/tensorflow-2.0-simple-examples
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

# load mnist
print('Loading MNIST dataset (please wait)...')
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print('MNIST dataset loaded')
x_train, x_test = x_train / 255.0, x_test / 255.0   # np.shape(x_train) = (60000,28,28) 

# define model
nn = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(200, activation='sigmoid'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# add optimizer, loss, and metric
nn.compile( optimizer='adam', 
            loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
            metrics=['accuracy'])

# train
nn.fit(x_train, y_train, epochs=5)

# Montre une image
plt.imshow(x_test[0], cmap='gray')
plt.show()

# Realise une prediction pour toutes les images du jeu de test
all_tests_predictions = nn.predict(x_test)

# Realise une prediction pour une seule image
first_test_prediction_onehot = nn.predict(np.array([ x_test[0] ]))
# Realise une prediction pour une seule image
first_test_prediction_integer = first_test_prediction_onehot.argmax()

print('La prediction est que cette image est un:', first_test_prediction_integer)
