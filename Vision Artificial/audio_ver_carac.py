# audio_explore.py
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import pandas as pd

from audio_pipeline import load_audio_dataset   # <-- usamos tu pipeline nuevo
from knn import KNN                             # <-- tu KNN propio

# ==== CONFIG ====
AUDIO_DIR       = r"C:\Users\gabri\OneDrive\Escritorio\Juano Uncuyo\IA\Proyecto Final\Audios"  # <-- carpeta RAÍZ que contiene subcarpetas por clase
K_LIST = [1, 3, 5, 7, 9]                # ks a evaluar (impares para evitar empates)
PCA_COMPONENTS =   2                    # 2 o 3
PCA_DIMS = [10,15,20,25,30]             # dims a probar con PCA(whiten=True)
TOPN = 10  
def loo_knn_accuracy(Xs, y, k):
    """Leave-One-Out con tu KNN (clase KNN de knn.py)."""
    y = np.asarray(y)
    n = len(y)
    preds = np.empty(n, dtype=object)
    for i in range(n):
        Xtr = np.delete(Xs, i, axis=0)
        ytr = np.delete(y,  i, axis=0)
        knn = KNN(k=k)
        knn.learning(Xtr, ytr)
        preds[i] = knn.predict(Xs[i])
    classes = np.unique(y)
    acc = (preds == y).mean()
    cm = confusion_matrix(y, preds, labels=classes)
    return acc, classes, cm

# --- Visualizaciones alternativas ---
def plot_lda_2d(Xs, y):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    import matplotlib.pyplot as plt
    lda = LDA(n_components=2)
    Z = lda.fit_transform(Xs, y)
    plt.figure(figsize=(9,7))
    for cls in np.unique(y):
        m = (y == cls)
        plt.scatter(Z[m,0], Z[m,1], s=60, alpha=0.85, edgecolor="k", linewidth=0.5, label=str(cls))
    plt.title("LDA 2D (supervisada)"); plt.legend(); plt.tight_layout(); plt.show()

def plot_tsne_2d(Xs, y, perplexity=20, random_state=42):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    Z = TSNE(n_components=2, perplexity=perplexity, learning_rate="auto",
             init="pca", random_state=random_state).fit_transform(Xs)
    plt.figure(figsize=(9,7))
    for cls in np.unique(y):
        m = (y == cls)
        plt.scatter(Z[m,0], Z[m,1], s=60, alpha=0.85, edgecolor="k", linewidth=0.5, label=str(cls))
    plt.title(f"t-SNE 2D (perplexity={perplexity})"); plt.legend(); plt.tight_layout(); plt.show()

def main1():
    # 1) Cargar dataset (extrae features con el pipeline nuevo)
    X, y = load_audio_dataset(AUDIO_DIR)
    if X.shape[0] == 0:
        print("[WARN] No hay audios en AUDIO_DIR o no se pudieron leer.")
        return
    print(f"Samples: {len(y)} | Dims: {X.shape[1]} | Clases: {Counter(y)}")

    # 2) Estandarizar
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    # 3) PCA para inspección visual
    plot_lda_2d(Xs, y)
    #plot_tsne_2d(Xs, y, perplexity=25)
    # 4) LOO accuracy para varios k
    print("\n== LOO accuracy por k ==")
    for k in K_LIST:
        acc, classes, cm = loo_knn_accuracy(Xs, y, k)
        print(f"k={k}: acc={acc*100:.1f}%  | clases={list(classes)}")
        head = " " * 10 + " ".join([f"{c:>12.12}" for c in classes])
        print(head)
        for i, c in enumerate(classes):
            row = " ".join([f"{v:12d}" for v in cm[i]])
            print(f"{c:>10.10} {row}")
        print("-"*60)
FEAT_NAMES = (
    [f"mfcc_{i}"   for i in range(1,14)] +
    [f"delta_{i}"  for i in range(1,14)] +
    [f"delta2_{i}" for i in range(1,14)] +
    ["centroid","bandwidth","rolloff","rms"]
)

def loo_acc(X, y, k):
    y = np.asarray(y); n=len(y); preds=np.empty(n, dtype=object)
    for i in range(n):
        Xtr = np.delete(X, i, axis=0); ytr = np.delete(y, i, axis=0)
        knn = KNN(k=k); knn.learning(Xtr, ytr)
        preds[i] = knn.predict(X[i])
    acc = (preds==y).mean()
    classes = np.unique(y); cm = confusion_matrix(y, preds, labels=classes)
    return acc, classes, cm

def fisher_scores(X, y):
    """Fisher score por feature: var_entre / var_intra."""
    X = np.asarray(X); y = np.asarray(y)
    classes = np.unique(y)
    mu_global = X.mean(axis=0)
    # Between-class
    num = np.zeros(X.shape[1])
    den = np.zeros(X.shape[1])
    for c in classes:
        Xc = X[y==c]
        wc = Xc.shape[0]
        mu_c = Xc.mean(axis=0)
        num += wc * (mu_c - mu_global)**2
        den += (Xc.var(axis=0, ddof=1) * max(wc-1,1))
    # Evitar /0
    den[den==0] = 1e-9
    return num/den

def heatmap_class_means(Xs, y):
    df = pd.DataFrame(Xs, columns=FEAT_NAMES)
    df["label"] = y
    M = df.groupby("label").mean().T  # (features x clases)
    plt.figure(figsize=(10, max(6, len(FEAT_NAMES)*0.25)))
    plt.imshow(M.values, aspect="auto")
    plt.yticks(range(M.shape[0]), M.index)
    plt.xticks(range(M.shape[1]), M.columns, rotation=0)
    plt.colorbar(label="media por clase (z-score)")
    plt.title("Heatmap: medias por clase (features estandarizadas)")
    plt.tight_layout(); plt.show()
    return M

def boxplots_top_features(Xs, y, feat_idx, labels):
    df = pd.DataFrame(Xs, columns=FEAT_NAMES); df["label"] = y
    n = len(feat_idx)
    cols = min(3, n); rows = int(np.ceil(n/cols))
    plt.figure(figsize=(4*cols, 3.2*rows))
    for i, j in enumerate(feat_idx, 1):
        plt.subplot(rows, cols, i)
        data = [df[df["label"]==lab][FEAT_NAMES[j]].values for lab in labels]
        plt.boxplot(data, labels=labels, showmeans=True)
        plt.title(FEAT_NAMES[j]); plt.xticks(rotation=15)
    plt.tight_layout(); plt.show()

def main():
    # 1) Cargar features crudas
    X, y = load_audio_dataset(AUDIO_DIR)
    if X.shape[0]==0:
        print("No hay audios."); return
    print(f"Samples: {len(y)} | Dims: {X.shape[1]} | Clases: {Counter(y)}")

    # 2) Estandarizar (igual que tu pipeline de KNN/PCA)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    # 3) Visualizar el "vector de características por clase"
    #M = heatmap_class_means(Xs, y)  # medias por clase en z-score

    # 4) Top features por discriminatividad (Fisher)
    F = fisher_scores(Xs, y)
    idx_sorted = np.argsort(F)[::-1]
    print("\nTop features por Fisher score:")
    for i in idx_sorted[:TOPN]:
        print(f"{FEAT_NAMES[i]:>12}: {F[i]:.3f}")
    # Barras
    #plt.figure(figsize=(8,4))
    #top_idx = idx_sorted[:TOPN]
    #plt.bar(range(TOPN), F[top_idx])
    #plt.xticks(range(TOPN), [FEAT_NAMES[j] for j in top_idx], rotation=45, ha="right")
    #plt.ylabel("Fisher score"); plt.title("Top features discriminativas")
    #plt.tight_layout(); plt.show()
    # Boxplots de esas top features
    labels = list(np.unique(y))
    #boxplots_top_features(Xs, y, top_idx[:6], labels)

    # 5) Baseline LOO con KNN estándar
    print("\n== LOO con z-score (baseline) ==")
    for k in K_LIST:
        acc, classes, cm = loo_acc(Xs, y, k)
        print(f"k={k}: acc={acc*100:.1f}% | clases={list(classes)}")
        # matriz compacta
        head = " " * 10 + " ".join([f"{c:>12.12}" for c in classes]); print(head)
        for i,c in enumerate(classes):
            print(f"{c:>10.10} " + " ".join([f"{v:12d}" for v in cm[i]]))
        print("-"*60)

    # 6) KNN con PCA + whitening (reduce correlación/ruido)
    print("\n== LOO con PCA(whiten=True) + KNN ==")
    for d in PCA_DIMS:
        pca = PCA(n_components=d, whiten=True, random_state=42).fit(Xs)
        Z = pca.transform(Xs)
        acc, _, _ = loo_acc(Z, y, k=5)
        print(f"PCA dims={d}: acc={acc*100:.1f}%")

    # 7) KNN ponderado por Fisher (re-escala columnas => distancia euclídea ponderada)
    w = F / (F.sum() + 1e-12)
    Xw = Xs * np.sqrt(w)  # equivalente a d^2 = sum(w_j (x_j - y_j)^2)
    print("\n== LOO con KNN ponderado por Fisher ==")
    for k in K_LIST:
        acc, _, _ = loo_acc(Xw, y, k)
        print(f"k={k}: acc={acc*100:.1f}%")

if __name__ == "__main__":
    main()
    main1()
