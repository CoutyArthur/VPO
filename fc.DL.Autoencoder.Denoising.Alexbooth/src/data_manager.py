import pickle
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path


# TODO use 1000 data points as validation set
# TODO delete after using to draw (or put in helper somewhere)

def standardize_dataset(x, axis=None):
    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.sqrt(((x - mean)**2).mean(axis=axis, keepdims=True))
    return (x - mean) / std

def add_gaussian_noise(X, mean=0, std=1):
    """Returns a copy of X with Gaussian noise."""
    return X.copy() + std * np.random.standard_normal(X.shape) + mean

class DataManager:
    def __init__(self):
        self.X32_train = None
        self.X32_val = None
        self.X64_train = None
        self.X64_val = None
        self.training_set_size = None
        self.load_data()

    def load_data(self):

        # Dossier où se trouve data_manager.py
        BASE_DIR = Path(__file__).parent

        # Dossier dataset (celeba)
        DATASET_DIR = BASE_DIR / "celeba"
        images32 = []
        images64 = []
        for image_path in DATASET_DIR.rglob("*.jpg"):
            img = Image.open(image_path.relative_to(BASE_DIR))
            img = img.convert('L')
            img64 = img.resize((64,64))
            img32 = img.resize((32,32))
            print(image_path.relative_to(BASE_DIR))
            images32.append(np.array(img32))
            images64.append(np.array(img64))
        celebaImg32 = np.array(images32)
        celebaImg64 = np.array(images64)

        celebaImg32 = celebaImg32.reshape(celebaImg32.shape[0], 32, 32, 1)
        celebaImg64 = celebaImg64.reshape(celebaImg64.shape[0], 64, 64, 1) 
    
        celebaImg32 = standardize_dataset(celebaImg32, axis=(1,2))
        celebaImg64 = standardize_dataset(celebaImg64, axis=(1,2))

        n_total = celebaImg32.shape[0]  # ici 100
        n_val = int(0.2 * n_total)    # 20% pour validation
        self.X32_train = celebaImg32[:-n_val]
        self.X32_val = celebaImg32[-n_val:]

        self.X64_train = celebaImg64[:-n_val]
        self.X64_val = celebaImg64[-n_val:]

        self.training_set_size = celebaImg32.shape[0]

    def get_batch(self, batch_size, use_resize=False,  train=False):
        """Returns tuple of (X, X_pred). Adds noise to X_pred if specified.
           Otherwise, X == X_pred.

           Arguments:
                batch_size: integer > 0
                use_noise: boolean

           Returns:
                Tuple of two numpy.ndarray of the same shape"""
        
        if train :
            indexes = np.random.randint(self.X32_train.shape[0], size=batch_size)
            if use_resize:
                return self.X32_train[indexes,:], self.X64_train[indexes,:]
            return self.X32_train[indexes,:], self.X32_train[indexes,:]
        else :
            indexes = np.random.randint(self.X32_val.shape[0], size=batch_size)
            if use_resize:
                return self.X32_val[indexes,:], self.X64_val[indexes,:]
            return self.X32_val[indexes,:], self.X32_val[indexes,:]
