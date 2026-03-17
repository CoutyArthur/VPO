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

from PIL import Image
from pathlib import Path

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MSE

def load_images():
    BASE_DIR = Path(__file__).parent
    # Dossier dataset (celeba)
    DATASET_DIR = BASE_DIR / "celeba"
    images = []
    for image_path in DATASET_DIR.rglob("*.jpg"):
        img = Image.open(image_path.relative_to(BASE_DIR))
        images.append(np.array(img))

    npImages = np.array(images)
    indexes = np.random.choice(images.shape[0], size=10, replace=False)
    
    return npImages[indexes,:]

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

    d_fake_history = []
    d_fake_avg_history = []

    plt.ion()  # mode interactif : mise à jour en temps réel
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    for epoch in range(n_epochs):
        print('Epoch', epoch, '/', n_epochs)
        
        for _ in tqdm(range(5)):

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

        d_fake_history.append(d_fake)
        d_fake_avg_history.append(np.mean(d_fake_history))

        print("Epoch {} - loss: {:.4f} | D(real): {:.3f} | D(fake): {:.3f}".format(epoch, loss, d_real, d_fake))
        noise = np.random.normal(0, 1, (1000, 10))
        generated = model_gen.predict(noise, verbose=0).flatten()
        X = generated.flatten()

        ax1.clear()
        ax1.hist(X, bins=40, density=True, alpha=0.7, label='Generated')
        ax1.plot(xs, stats.laplace.pdf(xs), 'r', label='Laplace target')
        ax1.set_title("Epoch {}".format(epoch))
        ax1.legend()

        ax2.clear()
        ax2.plot(d_fake_history, label='D(fake)', alpha=0.6)
        ax2.plot(d_fake_avg_history, label='Avg D(fake)', linewidth=2)
        ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Idéal (0.5)')
        ax2.set_title("D(fake) au fil des epochs")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("D(fake)")
        ax2.set_ylim(0, 1)
        ax2.legend()

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.1)

        if epoch > 20 :
            totalAvgD = 0
            totalD = 0
            for i in range(len(d_fake_history) - 10, len(d_fake_history)):
                totalAvgD += d_fake_avg_history[i]
                totalD += d_fake_history[i]
            print(totalAvgD)
            print(totalD)  
            if(totalAvgD+0.01 > totalD and totalAvgD-0.01<totalD):
                break

        

    print("Finished training.")
    plt.ioff()
    plt.show() 





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