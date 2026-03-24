
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

#Pour l'inférence choisi, z est à modifier ligne 266
flags.DEFINE_integer("show_mode", 0, "choix de mode entre interpolate = 0, inférence choisi = 1, inférence aléatoire = 2")
FLAGS = flags.FLAGS

MODEL_GEN_PATH  = "generator.h5"
MODEL_DISC_PATH = "discriminator.h5"

def load_images():
    BASE_DIR = Path(__file__).parent
    # Dossier dataset (celeba)
    DATASET_DIR = BASE_DIR / "celeba"
    images = []
    for image_path in DATASET_DIR.rglob("*.jpg"):
        img = Image.open(image_path.relative_to(BASE_DIR))
        img = img.resize((128,128)).convert("RGB")
        images.append(np.array(img))

    npImages = np.array(images)
    npImages = npImages.reshape(npImages.shape[0], 128, 128, 3) 
    npImages = npImages.astype(np.float32) / 127.5 - 1.0 
    indexes = np.random.choice(npImages.shape[0], size=10, replace=False)
    
    return npImages[indexes] 


IMG_H = 128
IMG_W = 128
IMG_DIM = IMG_H * IMG_W 
LATENT_DIM = 100

def make_generator():
    model = tf.keras.Sequential(name="generator")
 
    # Projection de l'espace latent vers un volume 2×2×512
    model.add(tf.keras.layers.Dense(2 * 2 * 512, use_bias=False, input_dim=LATENT_DIM))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Reshape((2, 2, 512)))

    # 2×2 → 4×4
    model.add(tf.keras.layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(0.2))
 
    # 4×4 → 8×8
    model.add(tf.keras.layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(0.2))
 
    # 8×8 → 16×16
    model.add(tf.keras.layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(0.2))
 
    # 16×16 → 32×32
    model.add(tf.keras.layers.Conv2DTranspose(32, kernel_size=4, strides=2, padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(0.2))
    # 32×32 → 64×64
    model.add(tf.keras.layers.Conv2DTranspose(16, kernel_size=4, strides=2, padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(0.2))
 
    # 64×64 → 128×128  (sortie finale)
    model.add(tf.keras.layers.Conv2DTranspose(3, kernel_size=4, strides=2, 
           padding='same', use_bias=False, activation='tanh'))  # sortie [-1, 1]
 
    # Vérification de la forme de sortie
    assert model.output_shape == (None, IMG_H, IMG_W, 3), \
        f"Forme de sortie inattendue : {model.output_shape}"
 
    return model


def make_discriminator():
    model = tf.keras.Sequential(name="discriminator")

    # 128×128 → 64×64
    model.add(tf.keras.layers.Conv2D(16, kernel_size=4, strides=2, padding='same',
                             input_shape=(IMG_H, IMG_W, 3)))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Dropout(0.5))
 
    # 64×64 → 32×32
    model.add(tf.keras.layers.Conv2D(32, kernel_size=4, strides=2, padding='same',
                             input_shape=(IMG_H, IMG_W, 1)))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Dropout(0.5))
 
    # 32×32 → 16×16
    model.add(tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, padding='same'))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Dropout(0.5))
 
    # 16×16 → 8×8
    model.add(tf.keras.layers.Conv2D(128, kernel_size=4, strides=2, padding='same'))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Dropout(0.5))
 
    # 8×8 → 4×4
    model.add(tf.keras.layers.Conv2D(256, kernel_size=4, strides=2, padding='same'))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Dropout(0.5))

    # # 4×4 → 2×2
    # model.add(tf.keras.layers.Conv2D(512, kernel_size=4, strides=2, padding='same'))
    # model.add(tf.keras.layers.LeakyReLU(0.2))
    # model.add(tf.keras.layers.Dropout(0.3))
 
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
 
    model.compile(optimizer=Adam(1e-4, beta_1=0.5),
                  loss='binary_crossentropy')
    return model

def make_gan(model_gen, model_disc):
    model_gan = tf.keras.Sequential()

    model_disc.trainable = False

    model_gan.add(model_gen)
    model_gan.add(model_disc)
    model_gan.compile(
    optimizer=Adam(2e-4, beta_1=0.5),  loss='binary_crossentropy')
    return model_gan

def augment_batch(batch):
    augmented = []
    for img in batch:
        if np.random.rand() > 0.5:
            img = img[:, ::-1, :]                          # flip horizontal
        img = img + np.random.uniform(-0.05, 0.05)         # jitter luminosité
        img = np.clip(img, -1.0, 1.0)
        augmented.append(img)
    return np.array(augmented)


def train(model_gen, model_disc, model_gan, real_images):

    n_epochs = 1000
    batch_size = 5
    n_real = len(real_images)
    nb_batch = n_real // batch_size

    d_fake_history = []
    d_fake_avg_history = []

    plt.ion()  # mode interactif : mise à jour en temps réel
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    

    for epoch in range(n_epochs):
        for _ in tqdm(range(nb_batch)):
            # Batch aléatoire d'images réelles
            idx = np.random.randint(0, n_real, batch_size)
            real_batch = augment_batch(real_images[idx])

            # Images générées
            noise = np.random.normal(0, 1, (batch_size, LATENT_DIM))
            fake_batch = model_gen.predict(noise, verbose=0)

            X = np.vstack((real_batch, fake_batch))
            y_real = np.ones((batch_size,1)) * 0.9
            y_fake = np.zeros((batch_size,1))

            y = np.vstack((y_real, y_fake))

            # --- Entraînement discriminateur ---
            model_disc.trainable = True
            noise_std = max(0.01, 0.05 * (1 - epoch / n_epochs))
            X_noisy = np.clip(X + np.random.normal(0, noise_std, X.shape), -1., 1.)
            loss_disc = model_disc.train_on_batch(X_noisy, y) 
            #loss_disc = model_disc.train_on_batch(X, y)

            # --- Entraînement générateur ---
            for _ in range(3):
                noise = np.random.normal(0, 1, (batch_size, LATENT_DIM))
                y_gen = np.ones((batch_size, 1))
                model_disc.trainable = False
                model_gan.train_on_batch(noise, y_gen)

            # --- Métriques D(real) et D(fake) ---
            d_real = model_disc.predict(real_batch, verbose=0).mean()
            d_fake = model_disc.predict(fake_batch, verbose=0).mean()
        
        
        
        d_fake_history.append(d_fake)
        d_fake_avg_history.append(np.mean(d_fake_history))
        

        ax.clear()
        ax.plot(d_fake_history, label='D(fake)', alpha=0.6)
        ax.plot(d_fake_avg_history, label='Avg D(fake)', linewidth=2)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Idéal (0.5)')
        ax.set_title("D(fake) au fil des epochs")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("D(fake)")
        ax.set_ylim(0, 1)
        ax.legend()

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.1)

        print("Epoch {:4d} | loss disc: {:.4f} | D(real): {:.3f} | D(fake): {:.3f}".format(epoch, loss_disc, d_real, d_fake))
        if (epoch + 1) % 25 == 0:  # toutes les 5 epochs
            model_gen.save(MODEL_GEN_PATH)
            model_disc.save(MODEL_DISC_PATH)
            print(f"  → Modèles sauvegardés (epoch {epoch+1})")
    model_gen.save(MODEL_GEN_PATH)
    model_disc.save(MODEL_DISC_PATH)
    print("Finished training.")
    plt.ioff()
    plt.show() 


def interpolate(model_gen):
    # Echantillonner deux points latents distincts
    z1 = np.random.normal(0, 1, (1, LATENT_DIM))
    z2 = np.random.normal(0, 1, (1, LATENT_DIM))

    img1 = model_gen.predict(z1, verbose=0)[0]
    img2 = model_gen.predict(z2, verbose=0)[0]

    # Point milieu dans l'espace latent
    z_mid = (z1 + z2) / 2.0
    img_mid = model_gen.predict(z_mid, verbose=0)[0]

    # Normalisation pour affichage
    def normalize(img):
        return (img - img.min()) / (img.max() - img.min() + 1e-8)

    # Affichage des 3 images : z1, milieu, z2
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    fig.suptitle("Interpolation dans l'espace latent", fontsize=14)

    axes[0].imshow(normalize(img1), vmin=0, vmax=1)
    axes[0].set_title("z1")
    axes[0].axis('off')

    axes[1].imshow(normalize(img_mid), vmin=0, vmax=1)
    axes[1].set_title("z_mid = (z1 + z2) / 2")
    axes[1].axis('off')

    axes[2].imshow(normalize(img2), vmin=0, vmax=1)
    axes[2].set_title("z2")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

def _show_single(img, title="Image générée"):
    """Affiche une seule image en niveaux de gris."""
    def normalize(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)
 
    plt.figure(figsize=(4, 4))
    plt.imshow(normalize(img), vmin=0, vmax=1)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def infer_random(model_gen):
    z = np.random.normal(0, 1, (1, LATENT_DIM))
    print("\n=== Z utilisé (copiez-collez dans Z_FIXE pour le rejouer) ===")
    # Affichage compact sur une ligne, facile à copier
    print("np.array([" + ", ".join(f"{v:.6f}" for v in z[0]) + "])")
    print("=" * 60)
 
    img = model_gen.predict(z, verbose=0)[0]
    _show_single(img, title=f"Inférence aléatoire\nZ[0]={z[0,0]:.3f}, Z[1]={z[0,1]:.3f}, ...")
 
 
def infer_fixed(model_gen, z_values):

    z = np.array(z_values, dtype=np.float32).reshape(1, LATENT_DIM)
    print(f"\n=== Z fixé utilisé (dim={z.shape[1]}) ===")
    print("np.array([" + ", ".join(f"{v:.6f}" for v in z[0]) + "])")
    print("=" * 60)
 
    img = model_gen.predict(z, verbose=0)[0]
    _show_single(img, title="Inférence Z fixé")

def main(argv):

    real_images = load_images() 

    # Charger les modèles existants ou en créer de nouveaux
    if Path(MODEL_GEN_PATH).exists() and Path(MODEL_DISC_PATH).exists():
        print("Chargement des modèles sauvegardés...")
        model_gen  = tf.keras.models.load_model(MODEL_GEN_PATH)
        model_disc = tf.keras.models.load_model(MODEL_DISC_PATH)
        print("Modèles chargés avec succès.")
    else:
        print("Aucun modèle trouvé, création de nouveaux modèles...")
        model_gen  = make_generator()
        model_disc = make_discriminator()
    model_gan  = make_gan(model_gen, model_disc)

    train(model_gen, model_disc, model_gan, real_images)
    if FLAGS.show_mode == 0:
        interpolate(model_gen)
    elif FLAGS.show_mode == 1:
        z = np.random.normal(0, 1, (1, LATENT_DIM))
        infer_fixed(model_gen, z)
    elif FLAGS.show_mode == 2:
        infer_random(model_gen)

if __name__ == '__main__':
    if not tf.test.is_gpu_available():
        print("WARNING: Not training with GPU. Training may be slow.")
    app.run(main)