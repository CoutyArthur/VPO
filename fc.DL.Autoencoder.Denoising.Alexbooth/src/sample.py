import os
import sys
import numpy as np
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from data_manager import DataManager

import matplotlib.pyplot as plt 
import cv2

from model import Encoder, Decoder

from absl import app
from absl import flags

flags.DEFINE_integer("sample_size", 10, "samples to test")
flags.DEFINE_string("model", "./trained_model/DAE-model-timestamp.h5", "Path to a trained model (.h5 file)")
flags.DEFINE_boolean("use_resize", True, "hyper resolution")
FLAGS = flags.FLAGS


def sample(model, n_samples):
    """Passes n random samples through the model and displays X & X_pred"""
    manager = DataManager()
    X, _ = manager.get_batch(n_samples, use_resize=FLAGS.use_resize, train=False)
    print("Input min/max:",X.min(), X.max())
    X_pred = model.predict(X)
    print("X_pred shape:", X_pred.shape)
    print("Output min/max:", X_pred.min(), X_pred.max())
    x_dim_32, y_dim_32 = X[0].shape[0], X[0].shape[1]
    x_dim_64, y_dim_64 = X_pred[0].shape[0], X_pred[0].shape[1]
    # X_32_up = np.array([cv2.resize(img, (64, 64)) for img in X[...,0]])
    # X_stitched = np.reshape(X_32_up.swapaxes(0,1), (64, 64 * n_samples))
    # X_pred_stitched = np.reshape(X_pred[...,0].swapaxes(0,1), (x_dim_64, y_dim_64*n_samples))
    # stitched_img = np.vstack((X_stitched, X_pred_stitched))
    # plt.imshow(stitched_img, cmap='gray')
    # plt.show()

    # --------
    # Fonction padding 32 -> 128
    # --------
    def pad_to_128(img32):
        padded = np.zeros((128, 128, 3))
        gray = img32[..., 0]
        rgb = np.stack([gray, gray, gray], axis=-1)
        padded[48:80, 48:80] = rgb
        return padded

    # --------
    # Rangée 1 : 32x32 avec padding
    # --------
    row1 = np.array([
        pad_to_128(img)
        for img in X
    ])

    # --------
    # Rangée 2 : Nearest neighbor
    # --------
    row2 = np.array([
        np.stack([cv2.resize(img[...,0], (128, 128), interpolation=cv2.INTER_NEAREST)] * 3, axis=-1)
        for img in X
    ])
    

    # --------
    # Rangée 3 : Bicubique
    # --------
    row3 = np.array([
        np.stack([cv2.resize(img[...,0], (128, 128), interpolation=cv2.INTER_CUBIC)] * 3, axis=-1)
        for img in X
    ])

    # --------
    # Rangée 4 : Deep learning
    # --------
    row4 = X_pred

    # Stitch horizontal
    def stitch_row(row):
        return np.concatenate(row, axis=1)

    stitched = np.vstack([
        stitch_row(row1),
        stitch_row(row2),
        stitch_row(row3),
        stitch_row(row4)
    ])

    plt.figure(figsize=(10,5))
    plt.imshow(stitched)
    plt.axis("off")
    plt.show()

def load_model():
    """Set up and return the model."""
    model_path = os.path.abspath(FLAGS.model)
    model = tf.keras.models.load_model(model_path)
    print("Loaded model output shape:", model.output_shape)

    # holds dimensions of latent vector once we find it
    z_dim = None

    # define encoder
    encoder_in  = tf.keras.Input(shape=(32, 32, 1))
    encoder_out = Encoder(encoder_in)
    encoder = tf.keras.Model(inputs=encoder_in, outputs=encoder_out)
 
    # load encoder weights and get the dimensions of the latent vector
    for i, layer in enumerate(model.layers):
        encoder.layers[i] = layer
        if layer.name == "encoder_output":
            z_dim = (layer.get_weights()[0].shape[-1])
            break

    # define encoder
    decoder_in  = tf.keras.Input(shape=(z_dim,))
    decoder_out = Decoder(decoder_in)
    decoder = tf.keras.Model(inputs=decoder_in, outputs=decoder_out)

    # load decoder weights
    found_decoder_weights = False
    decoder_layer_cnt = 0
    for i, layer in enumerate(model.layers):
        print(layer.name)
        weights = layer.get_weights()
        if len(layer.get_weights()) > 0:
            print(weights[0].shape, weights[1].shape)
        if "decoder_input" == layer.name:
            found_decoder_weights = True
        if found_decoder_weights:
            decoder_layer_cnt += 1
            print("dec:" + decoder.layers[decoder_layer_cnt].name)
            decoder.layers[decoder_layer_cnt].set_weights(weights)

    encoder.summary()
    decoder.summary()

    return encoder, decoder, model
       
def main(argv):
    if FLAGS.model == None:
        print("Please specify a path to a model with the --model flag")
        sys.exit()
    encoder, decoder, autoencoder = load_model()
    sample(autoencoder, FLAGS.sample_size)

if __name__ == '__main__':
    app.run(main)
