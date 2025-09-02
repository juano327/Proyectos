# ver_carac.py
# -----------------------------------------------------------------------------
# Visualización PCA del dataset + proyección de una imagen NUEVA en el mismo
# espacio (2D). Usa el pipeline de 'procesamiento.py' y parámetros de 'config.py'.
# Si hay artefactos de entrenamiento (mu/sigma), los usa para estandarizar.
# -----------------------------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from procesamiento2 import ProcessingPipeline, Standardizer
from config import SOURCE_DIR, DEST_DIR, ARTIF_DIR, OUTPUT_SIZE, HUE_BINS
try:
    from config import V_BINS
except ImportError:
    V_BINS = 8


# ------------------------ utilidades de Z-score -------------------------------

def zscore_with_artifacts_or_fallback(X: np.ndarray):
    """
    Si existen mu/sigma en ARTIF_DIR, los usa (consistente con el modelo).
    Si no, calcula un Z-score propio y avisa por consola.
    Devuelve: Xz, std (objeto Standardizer)
    """
    mu_path = os.path.join(ARTIF_DIR, "mu.npy")
    sigma_path = os.path.join(ARTIF_DIR, "sigma.npy")
    if os.path.isfile(mu_path) and os.path.isfile(sigma_path):
        std = Standardizer.load(ARTIF_DIR)
        Xz = std.transform(X)
        print("[INFO] Usando mu/sigma de artefactos para Z-score.")
        return Xz, std
    else:
        print("[WARN] No se hallaron artefactos (mu/sigma). Se hará Z-score propio.")
        std = Standardizer().fit(X)
        Xz = std.transform(X)
        return Xz, std


# ------------------------- PCA y gráficos ------------------------------------

def fit_pca_on_dataset(Xz: np.ndarray, n_components=2, random_state=42):
    pca = PCA(n_components=n_components, random_state=random_state)
    Z = pca.fit_transform(Xz)
    ev = pca.explained_variance_ratio_ * 100.0
    return pca, Z, ev

def plot_pca_dataset_with_new_point(Z: np.ndarray,
                                    y: np.ndarray,
                                    paths: list,
                                    Z_new: np.ndarray = None,
                                    label_new: str = None,
                                    annotate_names: bool = True,
                                    title: str = "PCA del dataset (con punto nuevo)"):
    """
    Dibuja el scatter PCA del dataset. Si Z_new está, dibuja también el punto nuevo.
    """
    classes = np.unique(y)
    markers = ['o','s','^','D','v','P','*','X']
    plt.figure(figsize=(9, 7))

    for i, c in enumerate(classes):
        idx = (y == c)
        plt.scatter(Z[idx, 0], Z[idx, 1],
                    s=50,
                    marker=markers[i % len(markers)],
                    edgecolor='black',
                    linewidth=0.5,
                    label=str(c))

    if annotate_names:
        for (x, ypt, p) in zip(Z[:, 0], Z[:, 1], paths):
            name = os.path.splitext(os.path.basename(p))[0]
            plt.text(x + 0.05, ypt + 0.05, name, fontsize=6)

    if Z_new is not None:
        # Z_new debe ser shape (2,) o (1,2)
        znew = np.array(Z_new).reshape(-1, 2)
        plt.scatter(znew[:, 0], znew[:, 1],
                    s=120, marker='X',
                    edgecolor='black', linewidth=1.0,
                    label=f"nueva → {label_new}" if label_new else "nueva")
        # pequeña flecha para destacarla (opcional)
        plt.annotate("nueva",
                     xy=(znew[0, 0], znew[0, 1]),
                     xytext=(znew[0, 0] + 0.5, znew[0, 1] + 0.5),
                     arrowprops=dict(arrowstyle="->", lw=1.2), fontsize=9)

    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title(title)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.show()


# ---------------------- API principal para usar desde la GUI ------------------

def plot_dataset_pca_and_project_new(image_path: str = None,
                                     predicted_label: str = None,
                                     annotate_names: bool = False):
    """
    - Procesa TODO el dataset (SOURCE_DIR) y calcula el PCA (2D).
    - Si image_path está definido:
        - Procesa la imagen NUEVA con el MISMO pipeline,
        - Estandariza con mu/sigma (artefactos si existen),
        - La proyecta en el PCA y la dibuja destacada.
    """
    # 1) Procesar dataset completo
    pipe = ProcessingPipeline(output_size=OUTPUT_SIZE, hue_bins=HUE_BINS, v_bins=V_BINS)
    X, y, paths = pipe.process_directory(SOURCE_DIR, DEST_DIR, mode="flat")
    if X.shape[0] == 0:
        print("[WARN] No hay imágenes para PCA.")
        return

    # 2) Z-score (idealmente con artefactos)
    Xz, std = zscore_with_artifacts_or_fallback(X)

    # 3) PCA del dataset
    pca, Z, ev = fit_pca_on_dataset(Xz, n_components=2, random_state=42)
    print(f"[INFO] Varianza explicada: PC1={ev[0]:.1f}%, PC2={ev[1]:.1f}%")

    # 4) Si hay imagen nueva, proyectarla
    Z_new = None
    if image_path:
        X_new, _ = pipe.process_one(image_path, DEST_DIR, label="desconocido")
        if X_new is None:
            print("[WARN] No se pudo procesar la imagen nueva para PCA.")
        else:
            Xz_new = std.transform(X_new.reshape(1, -1))  # Z-score con los mismos parámetros
            Z_new = pca.transform(Xz_new)[0]              # (2,)
            print(f"[INFO] Imagen nueva proyectada en PCA: {Z_new}")

    # 5) Dibujar
    plot_pca_dataset_with_new_point(
        Z=Z, y=y, paths=paths,
        Z_new=Z_new, label_new=predicted_label,
        annotate_names=annotate_names,
        title="PCA del dataset (punto nuevo resaltado)"
    )


# ----------------------------- CLI de prueba ----------------------------------

if __name__ == "__main__":
    # Ejemplo: solo dataset, sin imagen nueva
    plot_dataset_pca_and_project_new(image_path=None, predicted_label=None, annotate_names=True)
