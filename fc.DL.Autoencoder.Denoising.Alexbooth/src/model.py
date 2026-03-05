import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape,BatchNormalization,UpSampling2D,Activation,MaxPooling2D


def Conv(n_filters, filter_width):
    return Conv2D(n_filters, filter_width, 
                  strides=2, padding="same", activation="relu")

def Deconv(n_filters, filter_width):
    return Conv2DTranspose(n_filters, filter_width, 
                           strides=2, padding="same", activation=None)

def Encoder(inputs):
    # Bloc 1 : 32x32 -> 16x16
    X = Conv(32, 5)(inputs)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    #X = MaxPooling2D(2)(X)

    # Bloc 2 : 16x16 -> 8x8
    X = Conv(64, 5)(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    #X = MaxPooling2D(2)(X)
    
    # Bloc 3 : 8x8 -> 4x4
    X = Conv(128, 3)(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    #X = MaxPooling2D(2)(X)
   
    # Bloc 4 : 4x4 -> 2x2
    X = Conv(256, 3)(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    #X = MaxPooling2D(2)(X)
    
    X = Flatten()(X)
    Z = Dense(1024, activation="relu", name="encoder_output")(X)

    return Z


def Decoder(Z):
    # Reshape du vecteur latent en tenseur 2x2x256
    X = Reshape((2, 2, 256), name="decoder_input")(Z)  

    # Bloc 1 : 2x2 -> 4x4
    X = Deconv(128, 3)(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)

    # Bloc 2 : 4x4 -> 8x8
    X = Deconv(64, 3)(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)

    # Bloc 3 : 8x8 -> 16x16
    X = Deconv(32, 3)(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)

    # Bloc 4 : 16x16 -> 32x32
    X = Deconv(16, 3)(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)

    # Bloc 5 : 32x32 -> 64x64
    X = Deconv(8, 3)(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)

    # Bloc final : 64x64 -> 128x128
   
    X = Deconv(3,3)(X)
    print("Avant sigmoid shape:", X.shape) 
    X = Activation("sigmoid")(X)
    print("Après sigmoid shape:", X.shape)

    return X

def AutoEncoder():
    X = tf.keras.Input(shape=(32, 32, 1))
    Z = Encoder(X)
    X_pred = Decoder(Z)
    return tf.keras.Model(inputs=X, outputs=X_pred)

