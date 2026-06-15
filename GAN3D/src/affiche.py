import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# =============================================================================
# PARAMÈTRES OPTIONNELS
# =============================================================================
PADDING      = 1      # padding autour du voxel (0 = aucun)
VOXEL_ALPHA  = 0.9    # transparence des voxels (0.0 à 1.0)
VOXEL_COLOR  = "steelblue"
VOXEL_COLOR_REF = "tomato"
ELEV         = 30     # angle d'élévation de la caméra (degrés)
AZIM         = 45     # angle azimutal de la caméra (degrés)


# =============================================================================
# AFFICHAGE D'UN OBJET 3D SEUL (existant)
# =============================================================================
def display_voxel(voxel: np.ndarray, title: str = "Objet 3D") -> None:
    """Affiche un tableau voxel 3D avec matplotlib."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    colors = np.where(voxel, VOXEL_COLOR, "none")

    ax.voxels(
        voxel,
        facecolors=colors,
        edgecolor="k",
        linewidth=0.1,
        alpha=VOXEL_ALPHA,
    )

    ax.view_init(elev=ELEV, azim=AZIM)

    ax.set_title(f"{title}\n(voxels {voxel.shape[0]}x{voxel.shape[1]}x{voxel.shape[2]})", fontsize=13)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_xlim(0, voxel.shape[0])
    ax.set_ylim(0, voxel.shape[1])
    ax.set_zlim(0, voxel.shape[2])

    plt.tight_layout()
    plt.show()


# =============================================================================
# FIGURE COURBES D(fake) — créée une fois, mise à jour en boucle
# =============================================================================
def create_curves_figure():
    """Crée et retourne (fig, ax) pour la courbe D(fake)."""
    plt.ion()
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    fig.canvas.manager.set_window_title("Courbes D(fake)")
    return fig, ax


def update_curves_figure(fig, ax, d_fake_history, d_fake_avg_history, epoch, d_fake):
    """Met à jour la courbe D(fake) brute et moyennée."""
    ax.clear()
    ax.plot(d_fake_history,     label='D(fake)',     alpha=0.6, color='steelblue')
    ax.plot(d_fake_avg_history, label='Avg D(fake)', linewidth=2, color='orange')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Idéal (0.5)')
    ax.set_ylim(0, 1)
    ax.set_title(f"D(fake) — epoch {epoch}  [D(fake)={d_fake:.3f}]")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("D(fake)")
    ax.legend(loc='upper right')
    fig.canvas.draw()
    fig.canvas.flush_events()


# =============================================================================
# FIGURE RECONSTRUCTION 3D LIVE — créée une fois, mise à jour en boucle
# =============================================================================
def create_live3d_figure():
    """Crée et retourne (fig, ax) pour la reconstruction 3D live."""
    fig = plt.figure(figsize=(5, 5))
    fig.canvas.manager.set_window_title("Reconstruction 3D (live)")
    ax = fig.add_subplot(111, projection='3d')
    return fig, ax


def update_live3d_figure(fig, ax, voxel, epoch, img_h, img_w, img_d):
    """Met à jour la vue 3D live avec le voxel généré courant."""
    ax.clear()
    ax.voxels(voxel, facecolors=VOXEL_COLOR, edgecolor='k', linewidth=0.1, alpha=VOXEL_ALPHA)
    ax.set_xlim(0, img_h)
    ax.set_ylim(0, img_w)
    ax.set_zlim(0, img_d)
    ax.set_title(f"Epoch {epoch}")
    ax.view_init(elev=ELEV, azim=AZIM)
    fig.canvas.draw()
    fig.canvas.flush_events()


def finalize_training_figures(fig_curves, fig_3d):
    """Fige les figures live à la fin de l'entraînement."""
    plt.ioff()
    fig_curves.canvas.draw()
    fig_3d.canvas.draw()
    plt.pause(0.5)


# =============================================================================
# FIGURE COMPARAISON GAN <-> VOXEL RÉEL LE PLUS PROCHE (échantillonnage interactif)
# =============================================================================
def create_comparison_figure():
    """Crée et retourne la figure pour la boucle d'échantillonnage interactif."""
    fig = plt.figure(figsize=(12, 6))
    return fig


def update_comparison_figure(fig, gen_voxel, nearest_voxel, nearest_name, iou,
                              img_h, img_w, img_d):
    """Affiche côte à côte l'échantillon GAN et le voxel réel le plus proche."""
    fig.clf()

    ax_left = fig.add_subplot(1, 2, 1, projection='3d')
    ax_left.voxels(gen_voxel, facecolors=VOXEL_COLOR, edgecolor='k',
                   linewidth=0.1, alpha=VOXEL_ALPHA)
    ax_left.set_xlim(0, img_h)
    ax_left.set_ylim(0, img_w)
    ax_left.set_zlim(0, img_d)
    ax_left.set_title("Échantillon GAN")
    ax_left.view_init(elev=ELEV, azim=AZIM)

    ax_right = fig.add_subplot(1, 2, 2, projection='3d')
    ax_right.voxels(nearest_voxel > 0.5, facecolors=VOXEL_COLOR_REF, edgecolor='k',
                    linewidth=0.1, alpha=VOXEL_ALPHA)
    ax_right.set_xlim(0, img_h)
    ax_right.set_ylim(0, img_w)
    ax_right.set_zlim(0, img_d)
    ax_right.set_title(f"Plus proche réel : {nearest_name}\n(IoU={iou:.3f})")
    ax_right.view_init(elev=ELEV, azim=AZIM)

    fig.suptitle("GAN  ←→  Voxel réel le plus proche", fontsize=13)
    plt.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.01)