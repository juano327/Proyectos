# audio_pipeline.py
import os, tempfile, json
import numpy as np
import soundfile as sf
import sounddevice as sd
import librosa
from scipy.signal import butter, sosfilt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from knn import KNN
try:
    from config import AUDIO_SPLIT_TOP_DB, AUDIO_TARGET_DUR, AUDIO_BAND_LOW, AUDIO_BAND_HIGH
except ImportError:
    AUDIO_SPLIT_TOP_DB = 25
    AUDIO_TARGET_DUR   = 1.0
    AUDIO_BAND_LOW     = 80
    AUDIO_BAND_HIGH    = 6500

# ------------------------- utils de audio -------------------------

def _butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    from scipy.signal import butter
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos

def _apply_bandpass(y, sr, low=AUDIO_BAND_LOW, high=AUDIO_BAND_HIGH, order=5):
    sos = _butter_bandpass(low, high, sr, order=order)
    return sosfilt(sos, y)

def _normalize(y):
    y = y.astype(np.float32)
    m = np.max(np.abs(y)) if y.size else 1.0
    return y / (m + 1e-12)

def _remove_silence(y, sr, top_db=AUDIO_SPLIT_TOP_DB):
    intervals = librosa.effects.split(y, top_db=top_db)
    if len(intervals) == 0:
        return y
    return np.concatenate([y[s:e] for s, e in intervals])

# ------------------------- extracción de features -------------------------

def extract_features_from_array(y, sr):
    if y.ndim > 1:
        y = librosa.to_mono(y.T if y.shape[0] < y.shape[1] else y)

    y = _normalize(y)
    y = _remove_silence(y, sr, top_db=AUDIO_SPLIT_TOP_DB)  # ← umbral silencios
    y = _fix_duration(y, sr, target_dur=AUDIO_TARGET_DUR)  # ← duración fija
    y = _apply_bandpass(y, sr, low=AUDIO_BAND_LOW, high=AUDIO_BAND_HIGH)  # ← banda
    # features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    d1   = librosa.feature.delta(mfcc)
    d2   = librosa.feature.delta(mfcc, order=2)
    sc   = librosa.feature.spectral_centroid(y=y, sr=sr)
    sb   = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    ro   = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rms  = librosa.feature.rms(y=y)

    feats = np.concatenate([
        mfcc.mean(axis=1),        # 13
        d1.mean(axis=1),          # 13
        d2.mean(axis=1),          # 13
        [sc.mean(), sb.mean(), ro.mean(), rms.mean()]  # 4
    ]).astype(np.float32)

    return feats

def extract_features_from_file(path, target_sr=44100):
    y, sr = librosa.load(path, sr=None, mono=True)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return extract_features_from_array(y, sr)

# ------------------------- carga de dataset -------------------------

def load_audio_dataset(root_dir):
    """
    Estructura esperada:
      root_dir/
        berenjena/*.wav
        papa/*.wav
        zanahoria/*.wav
        camote/*.wav
    Devuelve X (N,43) y (N,)
    """
    X, y = [], []
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"No existe AUDIO_DIR: {root_dir}")

    for cls in sorted(os.listdir(root_dir)):
        cpath = os.path.join(root_dir, cls)
        if not os.path.isdir(cpath):
            continue
        for fn in os.listdir(cpath):
            if not fn.lower().endswith((".wav", ".flac", ".mp3", ".ogg", ".m4a")):
                continue
            apath = os.path.join(cpath, fn)
            try:
                feats = extract_features_from_file(apath)
                X.append(feats)
                y.append(cls)
            except Exception as e:
                print(f"[WARN] fallo '{apath}': {e}")

    if not X:
        return np.zeros((0,43), np.float32), np.array([])
    return np.stack(X).astype(np.float32), np.array(y)

# ------------------------- persistencia -------------------------

def _save_np(path, arr): os.makedirs(os.path.dirname(path), exist_ok=True); np.save(path, arr)

def save_artifacts_audio(artifact_dir, scaler, X_scaled, y, pca=None):
    os.makedirs(artifact_dir, exist_ok=True)
    _save_np(os.path.join(artifact_dir, "audio_mu.npy"),    scaler.mean_[None, :])
    _save_np(os.path.join(artifact_dir, "audio_sigma.npy"), scaler.scale_[None, :])
    _save_np(os.path.join(artifact_dir, "audio_X_scaled.npy"), X_scaled)
    _save_np(os.path.join(artifact_dir, "audio_y.npy"), y)
    meta = {"pca_n_components": int(getattr(pca, "n_components_", 0))}
    with open(os.path.join(artifact_dir, "audio_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    if pca is not None:
        _save_np(os.path.join(artifact_dir, "audio_pca_components.npy"), pca.components_)
        _save_np(os.path.join(artifact_dir, "audio_pca_mean.npy"), pca.mean_)
        _save_np(os.path.join(artifact_dir, "audio_pca_explvar.npy"), pca.explained_variance_ratio_)

def load_artifacts_audio(artifact_dir):
    mu    = np.load(os.path.join(artifact_dir, "audio_mu.npy"))
    sigma = np.load(os.path.join(artifact_dir, "audio_sigma.npy"))
    Xs    = np.load(os.path.join(artifact_dir, "audio_X_scaled.npy"))
    y     = np.load(os.path.join(artifact_dir, "audio_y.npy"), allow_pickle=True)
    scaler = StandardScaler()
    scaler.mean_ = mu[0]
    scaler.scale_ = sigma[0]
    scaler.var_ = scaler.scale_**2  # por compatibilidad
    return scaler, Xs, y
# --- Fisher & pesos ---
def fisher_scores(Xs, y):
    # Xs ya estandarizado (z-score). y = array de etiquetas (strings)
    import numpy as np
    y = np.asarray(y)
    classes = np.unique(y)
    mu_g = Xs.mean(axis=0)
    num = np.zeros(Xs.shape[1], dtype=np.float64)
    den = np.zeros(Xs.shape[1], dtype=np.float64)
    for c in classes:
        Xc = Xs[y == c]
        wc = Xc.shape[0]
        mu_c = Xc.mean(axis=0)
        num += wc * (mu_c - mu_g) ** 2           # varianza entre clases
        den += np.maximum(Xc.var(axis=0, ddof=1) * max(wc - 1, 1), 1e-12)  # intra
    return num / den

def save_weights_audio(artifact_dir, w):
    np.save(os.path.join(artifact_dir, "audio_weights.npy"), w.astype(np.float32))

def load_weights_audio(artifact_dir):
    path = os.path.join(artifact_dir, "audio_weights.npy")
    if os.path.isfile(path):
        return np.load(path)
    # si no hay pesos, usar todos 1
    return None
# --- Visualización: LDA del dataset de audio (2D o 3D) ---
# --- Visualización: LDA del dataset de audio (2D o 3D) ---
def plot_audio_lda_dataset(artifact_dir, weighted=True, n_components=2, title=None, show=True):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

    scaler, Xs, y = load_artifacts_audio(artifact_dir)
    try:
        w = load_weights_audio(artifact_dir)
    except Exception:
        w = None

    if weighted and (w is not None):
        Xw = Xs * np.sqrt(w, dtype=Xs.dtype)
        ttl = title or f"LDA {n_components}D (audio, ponderada)"
    else:
        Xw = Xs
        ttl = title or f"LDA {n_components}D (audio, z-score)"

    classes = np.unique(y)
    n_comp = int(min(n_components, max(1, len(classes) - 1)))
    if n_comp < 1:
        print("[WARN] LDA requiere >= 2 clases.");  return None

    lda = LDA(n_components=n_comp).fit(Xw, y)
    Z = lda.transform(Xw)

    fig = None
    if n_comp == 2:
        fig = plt.figure(figsize=(9, 7))
        for cls in classes:
            m = (y == cls)
            plt.scatter(Z[m,0], Z[m,1], s=60, alpha=0.85, edgecolor="k", linewidth=0.5, label=str(cls))
        plt.xlabel("LD1"); plt.ylabel("LD2"); plt.title(ttl); plt.legend(); plt.tight_layout()
    else:
        from mpl_toolkits.mplot3d import Axes3D  # noqa
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        for cls in classes:
            m = (y == cls)
            ax.scatter(Z[m,0], Z[m,1], Z[m,2], s=60, alpha=0.85, label=str(cls))
        ax.set_xlabel("LD1"); ax.set_ylabel("LD2"); ax.set_zlabel("LD3")
        ax.set_title(ttl); ax.legend(); plt.tight_layout()

    if show:
        try:
            import matplotlib.pyplot as plt
            plt.show(block=False)   # <<< NO bloquear el loop de Tk
        except Exception as e:
            print(f"[WARN] No se pudo mostrar la figura: {e}")
    return fig


# --- LDA del dataset + punto nuevo desde archivo ---
def plot_audio_lda_with_new_file(artifact_dir, new_audio_path, weighted=True,
                                 n_components=2, title=None, marker_size=180,
                                 pred_label=None, show=True):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

    scaler, Xs, y = load_artifacts_audio(artifact_dir)
    try:
        w = load_weights_audio(artifact_dir)
    except Exception:
        w = None

    if weighted and (w is not None):
        Xw = Xs * np.sqrt(w, dtype=Xs.dtype)
        ttl = title or f"LDA {n_components}D (audio, ponderada) + nuevo"
    else:
        Xw = Xs
        ttl = title or f"LDA {n_components}D (audio, z-score) + nuevo"

    feats = extract_features_from_file(new_audio_path)  # (43,)
    fz = scaler.transform(feats.reshape(1, -1))[0]
    if weighted and (w is not None):
        fz = fz * np.sqrt(w, dtype=fz.dtype)

    classes = np.unique(y)
    n_comp = int(min(n_components, max(1, len(classes) - 1)))
    if n_comp < 1:
        print("[WARN] LDA requiere >= 2 clases.");  return None
    lda = LDA(n_components=n_comp).fit(Xw, y)
    Z  = lda.transform(Xw)
    zN = lda.transform(fz.reshape(1, -1))[0]

    fig = None
    if n_comp == 2:
        fig = plt.figure(figsize=(9, 7))
        for cls in classes:
            m = (y == cls)
            plt.scatter(Z[m,0], Z[m,1], s=60, alpha=0.85, edgecolor="k", linewidth=0.5, label=str(cls))
        # punto nuevo
        plt.scatter(zN[0], zN[1], s=marker_size, marker="x", linewidths=3, color="black",
                    label=("nuevo" if pred_label is None else f"nuevo: {pred_label}"))
        if pred_label is not None:
            plt.annotate(str(pred_label), (zN[0], zN[1]), textcoords="offset points", xytext=(8,8))
        plt.xlabel("LD1"); plt.ylabel("LD2"); plt.title(ttl); plt.legend(); plt.tight_layout()
    else:
        from mpl_toolkits.mplot3d import Axes3D  # noqa
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        for cls in classes:
            m = (y == cls)
            ax.scatter(Z[m,0], Z[m,1], Z[m,2], s=60, alpha=0.85, label=str(cls))
        ax.scatter(zN[0], zN[1], zN[2], s=marker_size, marker="x", linewidths=3, color="black",
                   label=("nuevo" if pred_label is None else f"nuevo: {pred_label}"))
        ax.set_xlabel("LD1"); ax.set_ylabel("LD2"); ax.set_zlabel("LD3")
        ax.set_title(ttl); ax.legend(); plt.tight_layout()

    if show:
        try:
            import matplotlib.pyplot as plt
            plt.show(block=False)   # <<< NO bloquear Tk
        except Exception as e:
            print(f"[WARN] No se pudo mostrar la figura: {e}")
    return fig



# --- Grabar, predecir y plotear LDA con el punto nuevo ---
def record_predict_and_plot(artifact_dir, k=5, duration=2.5, sr=44100,
                            weighted=True, n_components=2):
    """
    Graba desde mic, guarda temporalmente, predice con KNN y dibuja LDA con el punto nuevo.
    Retorna la etiqueta predicha.
    """
    import tempfile, sounddevice as sd, soundfile as sf

    # 1) Grabar
    rec = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()

    # 2) Guardar temporal y predecir
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    sf.write(tmp_path, rec.flatten(), sr)

    pred = predict_audio_file(tmp_path, artifact_dir, k=k)

    # 3) Dibujar LDA con el nuevo punto
    plot_audio_lda_with_new_file(artifact_dir, tmp_path,
                                 weighted=weighted, n_components=n_components)

    return pred
# --- Detección simple de voz efectiva (para bloquear silencio) ---
def has_speech(y, sr, top_db=None, min_voiced_dur=0.25, min_rms=0.01):
    """
    Retorna (ok, dur_efectiva_seg, rms_total).
    ok=True si, tras remover silencios, hay al menos 'min_voiced_dur' segundos
    y el RMS global supera 'min_rms'.
    """
    import numpy as np
    import librosa

    if top_db is None:
        try:
            from config import AUDIO_SPLIT_TOP_DB as top_db
        except Exception:
            top_db = 25

    # normalizo por seguridad
    y = y.astype(np.float32, copy=False)
    m = np.max(np.abs(y)) + 1e-12
    y = y / m

    # quitar silencios y medir duración efectiva
    intervals = librosa.effects.split(y, top_db=top_db)
    if len(intervals) == 0:
        dur_eff = 0.0
        y_eff = np.zeros(0, dtype=y.dtype)
    else:
        y_eff = np.concatenate([y[s:e] for s, e in intervals])
        dur_eff = len(y_eff) / float(sr)

    rms = float(np.sqrt(np.mean(y**2))) if len(y) else 0.0
    ok = (dur_eff >= float(min_voiced_dur)) and (rms >= float(min_rms))
    return ok, dur_eff, rms

# ------------------------- entrenamiento KNN -------------------------

def train_audio_knn(audio_dir, artifact_dir, k=5, do_pca_plot=False,
                    pca_components=3, random_state=42, use_feature_weighting=True, weight_gamma=1.0):
    X, y = load_audio_dataset(audio_dir)
    if X.shape[0] == 0:
        print("[WARN] No hay audios para entrenar.")
        return 0.0, np.array([])

    # 1) Z-score
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    # 2) Pesos por Fisher (opcional)
    w = None
    if use_feature_weighting:
        F = fisher_scores(Xs, y)
        # normalizá y suavizá (gamma=1 mantiene, <1 aplana, >1 enfatiza)
        F[F < 0] = 0.0
        w = (F / (F.sum() + 1e-12)) ** weight_gamma
        # aplicamos sqrt(w) para que d^2 = sum( w_j (x_j - y_j)^2 )
        Xs_w = Xs * np.sqrt(w, dtype=Xs.dtype)
    else:
        Xs_w = Xs

    # 3) LOO rápido para ver training-acc
    knn = KNN(k=k); knn.learning(Xs_w, y)
    correct = 0
    for i in range(len(Xs_w)):
        Xtr = np.delete(Xs_w, i, axis=0); ytr = np.delete(y, i, axis=0)
        knn_lo = KNN(k=k); knn_lo.learning(Xtr, ytr)
        pred = knn_lo.predict(Xs_w[i])
        correct += int(pred == y[i])
    acc = correct / len(Xs_w)

    # 4) Guardar artefactos
    save_artifacts_audio(artifact_dir, scaler, Xs, y)  # mu, sigma, Xs, y
    if w is not None:
        save_weights_audio(artifact_dir, w)            # audio_weights.npy

    # 5) (opcional) PCA de Xs_w para graficar
    if do_pca_plot and Xs_w.shape[0] >= 2:
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
        if pca_components == 2:
            pca = PCA(n_components=2, random_state=random_state).fit(Xs_w)
            Z = pca.transform(Xs_w)
            plt.figure(figsize=(9,7))
            for cls in np.unique(y):
                m = (y == cls)
                plt.scatter(Z[m,0], Z[m,1], label=cls, alpha=0.75, edgecolor="k", linewidth=0.5)
            ev = pca.explained_variance_ratio_*100
            plt.xlabel(f"PC1 ({ev[0]:.1f}%)"); plt.ylabel(f"PC2 ({ev[1]:.1f}%)")
            plt.title("PCA de audios (2D, con pesos)"); plt.legend(); plt.tight_layout(); plt.show()
        else:
            pca = PCA(n_components=3, random_state=random_state).fit(Xs_w)
            Z = pca.transform(Xs_w)
            from mpl_toolkits.mplot3d import Axes3D  # noqa
            fig = plt.figure(figsize=(11,8))
            ax = fig.add_subplot(111, projection="3d")
            for cls in np.unique(y):
                m = (y == cls)
                ax.scatter(Z[m,0], Z[m,1], Z[m,2], label=cls, alpha=0.8)
            ev = pca.explained_variance_ratio_*100
            ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)"); ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)"); ax.set_zlabel(f"PC3 ({ev[2]:.1f}%)")
            ax.set_title("PCA de audios (3D, con pesos)"); ax.legend(); plt.tight_layout(); plt.show()

    return acc, np.unique(y)

# ------------------------- predicción -------------------------

def predict_audio_file(path, artifact_dir, k=5):
    scaler, Xs, y = load_artifacts_audio(artifact_dir)
    w = load_weights_audio(artifact_dir)  # puede ser None
    knn = KNN(k=k)
    # si hay pesos, entrenamos con Xs ponderado
    Xs_w = Xs * (np.sqrt(w, dtype=Xs.dtype) if w is not None else 1.0)
    knn.learning(Xs_w, y)

    feats = extract_features_from_file(path)
    fz = scaler.transform(feats.reshape(1,-1))[0]
    if w is not None:
        fz = fz * np.sqrt(w, dtype=fz.dtype)
    return knn.predict(fz)

def record_and_predict(artifact_dir, k=5, duration=2.5, sr=44100):
    print("Presioná ENTER para comenzar a grabar…"); input()
    print("Grabando…")
    rec = sd.rec(int(duration*sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    print("Grabación finalizada.")
    # guardar temporal por debug (wav)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        sf.write(tmp.name, rec.flatten(), sr)
        path = tmp.name
    pred = predict_audio_file(path, artifact_dir, k=k)
    try: os.remove(path)
    except: pass
    return pred

def _fix_duration(y, sr, target_dur=AUDIO_TARGET_DUR):
    target_len = int(round(target_dur * sr))
    if len(y) == 0:
        return np.zeros(target_len, dtype=np.float32)
    if len(y) > target_len:
        # recorte centrado
        start = (len(y) - target_len) // 2
        return y[start:start+target_len]
    if len(y) < target_len:
        # padding centrado con ceros
        pad = target_len - len(y)
        left = pad // 2
        right = pad - left
        return np.pad(y, (left, right), mode="constant")
    return y
# --- Visualización: LDA del dataset de audio (2D o 3D) ---
def plot_audio_lda_dataset(artifact_dir, weighted=True, n_components=2, title=None):
    """
    Dibuja LDA del dataset de audio usando los artefactos guardados.
    - weighted=True aplica los mismos pesos por Fisher (si existen) que usás en el KNN.
    - n_components=2 (o 3) para la proyección.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

    # Cargar X (z-score) e y del entrenamiento
    scaler, Xs, y = load_artifacts_audio(artifact_dir)
    # Cargar pesos (si existen)
    try:
        w = load_weights_audio(artifact_dir)
    except Exception:
        w = None

    # Aplicar ponderación si corresponde
    if weighted and (w is not None):
        Xw = Xs * np.sqrt(w, dtype=Xs.dtype)
        ttl = title or f"LDA {n_components}D (audio, ponderada)"
    else:
        Xw = Xs
        ttl = title or f"LDA {n_components}D (audio, z-score)"

    classes = np.unique(y)
    n_comp = int(min(n_components, max(1, len(classes) - 1)))
    if n_comp < 1:
        print("[WARN] LDA requiere >= 2 clases.")
        return

    lda = LDA(n_components=n_comp).fit(Xw, y)
    Z = lda.transform(Xw)

    if n_comp == 2:
        plt.figure(figsize=(9, 7))
        for cls in classes:
            m = (y == cls)
            plt.scatter(Z[m, 0], Z[m, 1], s=60, alpha=0.85, edgecolor="k", linewidth=0.5, label=str(cls))
        plt.xlabel("LD1"); plt.ylabel("LD2")
        plt.title(ttl); plt.legend(); plt.tight_layout(); plt.show()
    else:
        from mpl_toolkits.mplot3d import Axes3D  # noqa
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        for cls in classes:
            m = (y == cls)
            ax.scatter(Z[m, 0], Z[m, 1], Z[m, 2], s=60, alpha=0.85, label=str(cls))
        ax.set_xlabel("LD1"); ax.set_ylabel("LD2"); ax.set_zlabel("LD3")
        ax.set_title(ttl); ax.legend(); plt.tight_layout(); plt.show()
