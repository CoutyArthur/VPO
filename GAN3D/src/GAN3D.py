import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import affiche

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tqdm import tqdm
import time 
from absl import app, flags
from pathlib import Path
from scipy.ndimage import label
from tensorflow.keras.optimizers import Adam

flags.DEFINE_integer("show_mode", 0, "0=interpolate, 1=inférence fixe, 2=inférence aléatoire")

FLAGS = flags.FLAGS

# =============================================================================
# PARAMÈTRES PRINCIPAUX & DIMENSIONS
# =============================================================================

SINGLE_OBJECT_MODE = True
OBJECT_NAME = "airplane"
ALL_OBJECTS = ["airplane", "bathtub", "bed", "bench", "bookshelf"]

IMG_H, IMG_W, IMG_D = 32, 32, 32
LATENT_DIM = 200
PADDING = 1

# =============================================================================
# HYPERPARAMÈTRES D'ENTRAÎNEMENT & CONVERGENCE
# =============================================================================

STEPS_PER_EPOCH  = 10
BATCH_SIZE       = 128
N_EPOCHS_MAX     = 1000
MIN_EPOCHS       = 30
CONV_WINDOW      = 10
CONV_PATIENCE    = 10
CONV_MEAN_TOL    = 0.08
CONV_STD_TOL     = 0.04
VISU_EVERY       = 10

LR_DISC = 2e-4
LR_GAN  = 2e-4

# =============================================================================
# CHARGEMENT DES DONNÉES
# =============================================================================

def _load_mat_files(object_name: str, split: str = "train") -> list:
    BASE_DIR    = Path(__file__).parent.parent
    DATASET_DIR = BASE_DIR / "data" / object_name / "30" / split
    mat_files   = sorted(DATASET_DIR.glob("*.mat"))
    if not mat_files:
        print(f"[ERREUR] Aucun fichier .mat dans {DATASET_DIR}")
    return mat_files


def _mat_to_voxel(path: Path, normalise: bool = True) -> np.ndarray:
    """Charge un .mat → tableau float32 (32,32,32) """
    mat   = sio.loadmat(str(path))
    voxel = mat["instance"].astype(np.float32)
    if PADDING > 0:
        voxel = np.pad(voxel, pad_width=PADDING, mode="constant", constant_values=0.0)
    return voxel * 2.0 - 1.0


def load_voxel(object_name: str = None) -> np.ndarray:
    name  = object_name or OBJECT_NAME
    files = _load_mat_files(name, split="test")
    if not files:
        return None
    mat   = sio.loadmat(str(files[0]))
    voxel = mat["instance"].astype(np.float32)  # Converti en float32 pour cohérence
    if PADDING > 0:
        voxel = np.pad(voxel, pad_width=PADDING, mode="constant", constant_values=0.0)
    return voxel


def load_all_voxels(object_names: list = None) -> tuple:
    names  = object_names or ([OBJECT_NAME] if SINGLE_OBJECT_MODE else ALL_OBJECTS)
    voxels, labels = [], []
    for name in names:
        files = _load_mat_files(name, split="train")
        for path in files:
            voxel = _mat_to_voxel(path, normalise=True)
            if voxel.ndim != 3:
                continue
            voxels.append(voxel)
            labels.append(name)
    arr = np.array(voxels)[..., np.newaxis]
    print(f"Dataset chargé : {arr.shape} objets={names}")
    return arr, labels


def load_reference_voxels(object_names: list = None) -> tuple:
    names = object_names or ([OBJECT_NAME] if SINGLE_OBJECT_MODE else ALL_OBJECTS)
    ref_voxels, ref_names = [], []
    for name in names:
        # Charger PLUSIEURS fichiers test, pas juste files[0]
        files = _load_mat_files(name, split="test")
        for path in files:
            mat   = sio.loadmat(str(path))
            voxel = mat["instance"].astype(np.float32)
            if PADDING > 0:
                voxel = np.pad(voxel, PADDING, mode="constant")
            ref_voxels.append(voxel)
            ref_names.append(name)
    return ref_voxels, ref_names


# =============================================================================
# ARCHITECTURES DES MODÈLES
# =============================================================================

def make_generator() -> tf.keras.Model:

    # 1 -> 2
    model = tf.keras.Sequential(name="generator")
    model.add(tf.keras.layers.Dense(2*2*2*512, use_bias=False, input_dim=LATENT_DIM))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Reshape((2, 2, 2, 512)))

    # 2 -> 4
    model.add(tf.keras.layers.Conv3DTranspose(256, kernel_size=4, strides=2, padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Dropout(0.1))

    # 4 -> 8
    model.add(tf.keras.layers.Conv3DTranspose(128, kernel_size=4, strides=2, padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Dropout(0.1))

    # 8 -> 16
    model.add(tf.keras.layers.Conv3DTranspose(64, kernel_size=4, strides=2, padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    # 16 -> 32
    model.add(tf.keras.layers.Conv3DTranspose(1, kernel_size=4, strides=2, padding='same', use_bias=False, activation='tanh'))
    return model



def make_discriminator() -> tf.keras.Model:

    model = tf.keras.Sequential(name="discriminator")

    # 32 → 16  (pas de BN sur la première couche — standard DCGAN)
    model.add(tf.keras.layers.Conv3D(64, kernel_size=4, strides=2, padding='same', input_shape=(IMG_H, IMG_W, IMG_D, 1)))
    model.add(tf.keras.layers.LeakyReLU(0.3))

    # 16 → 8
    model.add(tf.keras.layers.Conv3D(128, kernel_size=4, strides=2, padding='same'))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Dropout(0.3))

    # # 8 → 4
    # model.add(tf.keras.layers.Conv3D(256, kernel_size=4, strides=2, padding='same'))
    # model.add(tf.keras.layers.LeakyReLU(0.2))
    # model.add(tf.keras.layers.Dropout(0.3))

    # # 4 → 2
    # model.add(tf.keras.layers.Conv3D(512, kernel_size=4, strides=2, padding='same'))
    # model.add(tf.keras.layers.LeakyReLU(0.2))
    # model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model



# =============================================================================
# NETTOYAGE & GÉNÉRATION
# =============================================================================

def _remove_noise(voxel: np.ndarray, min_size: int = 4) -> np.ndarray:
    labeled, n = label(voxel)
    cleaned = np.zeros_like(voxel)
    for i in range(1, n + 1):
        if (labeled == i).sum() >= min_size:
            cleaned |= (labeled == i)
    return cleaned



def generate_voxel(model_gen, z: np.ndarray = None, threshold: float = 0.0) -> np.ndarray:
    """Seuil à 0.5 car la sortie du générateur est une Sigmoid [0, 1]."""
    if z is None:
        z = np.random.normal(0, 1.5, (1, LATENT_DIM))
    out   = model_gen.predict(z, verbose=0)[0, ..., 0]
    voxel = out > threshold
    return _remove_noise(voxel), z


# =============================================================================
# CONVERGENCE
# =============================================================================

def _has_converged(d_fake_history: list, epoch: int) -> bool:
    if epoch < MIN_EPOCHS or len(d_fake_history) < CONV_WINDOW:
        return False
    consecutive = 0
    for i in range(len(d_fake_history) - 1, max(-1, len(d_fake_history) - CONV_PATIENCE - CONV_WINDOW), -1):
        window = d_fake_history[max(0, i - CONV_WINDOW + 1): i + 1]
        if len(window) < CONV_WINDOW:
            break
        m = abs(np.mean(window) - 0.5)
        s = np.std(window)
        if m < CONV_MEAN_TOL and s < CONV_STD_TOL:
            consecutive += 1
        else:
            break

    return consecutive >= CONV_PATIENCE



def train(model_gen, model_disc, model_disc_frozen,  model_gan, real_voxels: np.ndarray):

    n_real = len(real_voxels)
    d_fake_history, d_fake_avg_history = [], []

    plt.ion()
    fig_curves, ax_curves = affiche.create_curves_figure()
    fig_3d, ax_3d = affiche.create_live3d_figure()

    # Nombre de batches par époque
    steps_per_epoch = max(1, n_real // BATCH_SIZE)

    t_start = time.time()

    for epoch in range(N_EPOCHS_MAX):
        for step in tqdm(range(STEPS_PER_EPOCH), desc=f"Epoch {epoch:4d}", ncols=80):
            idx        = np.random.choice(n_real, BATCH_SIZE, replace=True)
            real_batch = real_voxels[idx]
            noise      = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
            fake_batch = model_gen(noise, training=False).numpy()

            y_real = np.random.uniform(0.7, 1.0, (BATCH_SIZE, 1))
            y_fake = np.random.uniform(0.0, 0.15, (BATCH_SIZE, 1))

            loss_d_real = model_disc.train_on_batch(real_batch, y_real)
            loss_d_fake = model_disc.train_on_batch(fake_batch, y_fake)

            # Resynchroniser la copie figée AVANT d'entraîner le générateur
            model_disc_frozen.set_weights(model_disc.get_weights())

            # Entraîner le générateur via model_gan (disc_frozen inchangé)
            for _ in range(1):
                noise_g = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
                loss_g  = model_gan.train_on_batch(noise_g, np.ones((BATCH_SIZE, 1)))

        # Métriques UNE SEULE FOIS par époque, après la boucle
        noise_m    = np.random.normal(0, 1, (16, LATENT_DIM))
        fake_m     = model_gen(noise_m, training=False).numpy()
        idx_m      = np.random.choice(n_real, 16, replace=True)
        real_m     = real_voxels[idx_m]

        d_real = model_disc(real_m,  training=False).numpy().mean()
        d_fake = model_disc(fake_m,  training=False).numpy().mean()

        d_fake_history.append(d_fake)
        window = d_fake_history[-20:] if len(d_fake_history) >= 20 else d_fake_history
        d_fake_avg_history.append(np.mean(window))

        loss_d = (loss_d_real + loss_d_fake) / 2.0
        print(f"Epoch {epoch:4d} | "
              f"loss_d={loss_d:.4f}  loss_g={loss_g:.4f} | "
              f"D(real)={d_real:.3f}  D(fake)={d_fake:.3f}")

        # ── Courbe D(fake) ────────────────────────────────────────────────────
        affiche.update_curves_figure(fig_curves, ax_curves, d_fake_history, d_fake_avg_history, epoch, d_fake)

        # ── Visualisation 3D live ─────────────────────────────────────────────
        if epoch % VISU_EVERY == 0:
            voxel_live, _ = generate_voxel(model_gen)
            affiche.update_live3d_figure(fig_3d, ax_3d, voxel_live, epoch, IMG_H, IMG_W, IMG_D)

        # ── Critère de convergence ────────────────────────────────────────────
        if _has_converged(d_fake_history, epoch):
            print(f"\n✓ Convergence à l'epoch {epoch}.")
            break

    t_end = time.time()
    duree = t_end - t_start
    minutes  = int((duree % 3600) // 60)
    secondes = int(duree % 60)
    print(f"─── Entraînement terminé en {minutes:02d}m{secondes:02d}s ")
    affiche.finalize_training_figures(fig_curves, fig_3d)

# =============================================================================
# EVALUATION & BOUCLE INTERACTIVE
# =============================================================================

def voxel_iou(a: np.ndarray, b: np.ndarray) -> float:
    """
    a : voxel généré  (sortie tanh, valeurs dans [-1, 1]) → seuil à 0.0
    b : voxel réel    (chargé par load_voxel, valeurs dans [0, 1]) → seuil à 0.5
    """
    a_bin = a > 0.0
    b_bin = b > 0.5
    inter = np.logical_and(a_bin, b_bin).sum()
    union = np.logical_or(a_bin, b_bin).sum()
    return float(inter) / (float(union) + 1e-8)



def find_nearest_voxel(generated: np.ndarray, ref_voxels: list, ref_names: list) -> tuple:

    best_iou, best_v, best_n = -1, None, None
    for v, n in zip(ref_voxels, ref_names):
        iou = voxel_iou(generated, v)
        if iou > best_iou:
            best_iou, best_v, best_n = iou, v, n
    return best_v, best_n, best_iou


def show_sample_loop(model_gen, ref_voxels: list, ref_names: list):

    print("\n─────────────────────────────────────────────")
    print("  Appuyez sur [ENTRÉE] pour un nouvel échantillon.")
    print("  Appuyez sur [Ctrl+C] puis [ENTRÉE] pour quitter.")
    print("─────────────────────────────────────────────\n")

    fig = affiche.create_comparison_figure()

    while True:
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            print("Fin.")
            break

        gen_voxel, z = generate_voxel(model_gen)
        nearest_voxel, nearest_name, iou = find_nearest_voxel(gen_voxel, ref_voxels, ref_names)
        affiche.update_comparison_figure(fig, gen_voxel, nearest_voxel, nearest_name, iou, IMG_H, IMG_W, IMG_D)


def main(argv):

    mode_str = "SINGLE" if SINGLE_OBJECT_MODE else "MULTI (5 objets)"
    print(f"\n{'='*50}")
    if (SINGLE_OBJECT_MODE) :
        print(f"  Mode : {mode_str}    Objet : {OBJECT_NAME}")
    else :
        print(f"  Mode : {mode_str}    Objets : {ALL_OBJECTS}")
    print(f"{'='*50}\n")

    # Données
    real_voxels, _ = load_all_voxels()
    ref_voxels, ref_names = load_reference_voxels()

    model_gen  = make_generator()

    model_disc = make_discriminator()
    model_disc.compile(optimizer=Adam(LR_DISC, beta_1=0.5), loss='binary_crossentropy')

    # Copie figée : jamais entraînée directement, elle ne sert qu'à fournir
    # le gradient vers le générateur dans model_gan.
    model_disc_frozen = make_discriminator()
    model_disc_frozen.set_weights(model_disc.get_weights())
    model_disc_frozen.trainable = False

    model_gan = tf.keras.Sequential([model_gen, model_disc_frozen], name="gan")
    model_gan.compile(optimizer=Adam(LR_GAN, beta_1=0.5), loss='binary_crossentropy')

    train(model_gen, model_disc, model_disc_frozen, model_gan, real_voxels)

    # Échantillonnage interactif
    show_sample_loop(model_gen, ref_voxels, ref_names)

if __name__ == '__main__':
    app.run(main)