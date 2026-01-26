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

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy() 

optimizer = tf.keras.optimizers.Adam(0.001)

def train(model, loss_fn, x_train, y_train):
    with tf.GradientTape() as g:
        predictions = model(x_train, training=True)
        loss = loss_fn(y_train, predictions)
    gradients = g.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

train_loss = tf.keras.metrics.Mean(name='train_loss')

# fit
for epoch in range(50):
    loss = train(nn, loss_fn, x_train, y_train)
    # Accumule le cout dans la metrique (la metrique est une moyenne)
    train_loss(loss)
    print('Accumulation du loss moyen pour ce batch', loss.numpy(), 'dans la metrique de moyenne. La moyenne a ce stade est', train_loss.result().numpy())

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
