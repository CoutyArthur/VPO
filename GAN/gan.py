import os
import sys
import glob
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tqdm import tqdm

from absl import app
from absl import flags

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MSE


def make_generator():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(15, activation='relu', input_dim=10))
    model.add(tf.keras.layers.Dense(1))
    return model

def make_discriminator():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(25, activation='relu', input_dim=1))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss=tf.keras.losses.BinaryCrossentropy())
    return model

def make_gan(model_gen, model_disc):
    model_gan = tf.keras.Sequential()

    model_disc.trainable = False

    model_gan.add(model_gen)
    model_gan.add(model_disc)
    model_gan.compile(loss=tf.keras.losses.BinaryCrossentropy())
    return model_gan

def train(model_gen, model_disc, model_gan):

    n_epochs = 200
    batch_size = 2000

    xs = np.linspace(-5, 5, 500)

    plt.ion()  # mode interactif : mise à jour en temps réel
    fig, ax = plt.subplots()

    for epoch in range(n_epochs):
        print('Epoch', epoch, '/', n_epochs)
        
        for _ in tqdm(range(1)):

            real_data = np.random.laplace(0, 1, batch_size).reshape(-1,1)
            noise = np.random.normal(0,1,(batch_size,10))
            fake_data = model_gen.predict(noise, verbose=0)

            X = np.vstack((real_data, fake_data))
            y = np.vstack((np.ones((batch_size,1)), np.zeros((batch_size,1))))

            model_disc.trainable = True
            loss = model_disc.train_on_batch(X, y)

            noise = np.random.normal(0,1,(batch_size,10))
            y_gen = np.ones((batch_size,1))

            model_disc.trainable = False
            model_gan.train_on_batch(noise, y_gen)

        noise = np.random.normal(0, 1, (200, 10))
        fake_samples = model_gen.predict(noise, verbose=0)
        real_samples = np.random.laplace(0, 1, 200).reshape(-1, 1)

        d_real = model_disc.predict(real_samples, verbose=0).mean()
        d_fake = model_disc.predict(fake_samples, verbose=0).mean()

        print("Epoch {} - loss: {:.4f} | D(real): {:.3f} | D(fake): {:.3f}".format(epoch, loss, d_real, d_fake))
        noise = np.random.normal(0, 1, (1000, 10))
        generated = model_gen.predict(noise, verbose=0).flatten()
        X = generated.flatten()

        ax.clear()
        ax.hist(X, bins=40, density=True, alpha=0.7, label='Generated')
        ax.plot(xs, stats.laplace.pdf(xs), 'r', label='Laplace target')
        ax.set_title("Epoch {}".format(epoch))
        ax.legend()

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.1)
    print("Finished training.")





def main(argv):

    model_gen = make_generator()
    model_disc = make_discriminator()

    model_gan = make_gan(model_gen, model_disc)

    train(model_gen, model_disc, model_gan)

    noise = np.random.normal(0,1,(1000,10))
    generated = model_gen.predict(noise)

    X = generated.flatten()

    plt.hist(X, 40, density=True)

    xs = np.linspace(-5,5,500)
    plt.plot(xs, stats.laplace.pdf(xs), 'r')

    plt.show()

if __name__ == '__main__':
    if not tf.test.is_gpu_available():
        print("WARNING: Not training with GPU. Training may be slow.")
    app.run(main)