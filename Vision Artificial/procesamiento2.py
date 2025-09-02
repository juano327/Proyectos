# procesamiento.py
# -----------------------------------------------------------------------------
# Pipeline completo de procesamiento + entrenamiento K-Means + clasificación.
# Lee parámetros (OUTPUT_SIZE, HUE_BINS, V_BINS) desde config.py.
# Usa TU implementación de KMeans en kmeans.py (init='seeded'|'kmeans++'|'random').
# -----------------------------------------------------------------------------

import os, re, json
from typing import List, Tuple
from collections import Counter

import cv2
import numpy as np

# PIL + pillow-heif para soportar HEIC/HEIF (en Windows/iPhone)
from PIL import Image
import pillow_heif
from kmeans import KMeans

# -----------------------------------------------------------------------------
# 1) Cargar parámetros desde config.py (con fallback a valores por defecto)
# -----------------------------------------------------------------------------
try:
    # config.py debe definir al menos: OUTPUT_SIZE, HUE_BINS
    from config import OUTPUT_SIZE, HUE_BINS
    try:
        # V_BINS es opcional; si no existe en config, usamos 8
        from config import V_BINS
    except ImportError:
        V_BINS = 8
except ImportError:
    # Valores por defecto si no hay config.py
    OUTPUT_SIZE = 256   # tamaño del recorte cuadrado final
    HUE_BINS    = 12    # cantidad de bins para histograma de H (0..180)
    V_BINS      = 8     # cantidad de bins para histograma de V (0..255)

# Extensiones de imagen soportadas
VALID_EXTS = (".png", ".jpg", ".jpeg", ".heic", ".heif",
              ".PNG", ".JPG", ".JPEG", ".HEIC", ".HEIF")


# -----------------------------------------------------------------------------
# 2) Utilidades de E/S
def safe_imread(path: str):
    """
    Lee una imagen desde 'path' soportando PNG/JPG/HEIC.
    - Para PNG/JPG usa OpenCV (BGR).
    - Para HEIC/HEIF usa Pillow + pillow-heif y lo convierte a BGR.
    Devuelve: ndarray BGR (H,W,3) o None si falla.
    """
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in (".png", ".jpg", ".jpeg"):
            # cv2.imread devuelve BGR por defecto
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            return img
        elif ext in (".heic", ".heif"):
            # pillow-heif permite abrir HEIC y convertir a RGB; luego pasamos a BGR para OpenCV
            pillow_heif.register_heif_opener()
            img_pil = Image.open(path).convert("RGB")
            img_np = np.array(img_pil)                         # RGB uint8
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) # BGR uint8
            return img_bgr
        else:
            print(f"[WARN] Formato no soportado: {ext} -> {path}")
            return None
    except Exception as e:
        print(f"[ERROR] No se pudo leer {path}: {e}")
        return None


def infer_label_from_filename(filename: str) -> str:
    """
    Infere la etiqueta a partir del nombre del archivo.
    - Si tiene formato 'etiqueta_loquesea.jpg' devuelve 'etiqueta'.
    - Si no, toma el prefijo alfabético (ej. 'zanahoria12' -> 'zanahoria').
    Devuelve en minúsculas.
    """
    base = os.path.basename(filename)
    name, _ = os.path.splitext(base)
    if "_" in name:
        return name.split("_")[0].strip().lower()
    m = re.match(r"([a-zA-ZñÑáéíóúÁÉÍÓÚ]+)", name)
    return m.group(1).lower() if m else name.lower()


def list_images(folder: str) -> List[Tuple[str, str]]:
    """
    Recorre recursivamente 'folder' y devuelve una lista de (etiqueta, ruta).
    La etiqueta se infiere con infer_label_from_filename().
    """
    items: List[Tuple[str, str]] = []
    for root, _, files in os.walk(folder):
        for fn in files:
            if fn.endswith(VALID_EXTS):
                items.append((infer_label_from_filename(fn), os.path.join(root, fn)))
    return items


# -----------------------------------------------------------------------------
# 3) Segmentación: Otsu sobre S (HSV) + morfología + componente más grande
# -----------------------------------------------------------------------------
class Segmenter:
    def __init__(self, morph_kernel: int = 5):
        # Kernel elíptico para operaciones morfológicas
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))

    def segment(self, bgr: np.ndarray) -> np.ndarray:
        """
        Devuelve una máscara binaria 0/255 del objeto principal.
        Pasos:
          - Convertir a HSV
          - Umbralizar el canal S con Otsu (resalta zonas saturadas -> objeto)
          - Morfología (close + open) para limpiar
          - Quedarse con el componente conexo de mayor área
        """
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        s = hsv[..., 1]  # canal de saturación

        # Umbral automático de Otsu (binario)
        _, m = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Limpieza morfológica
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, self.kernel, iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  self.kernel, iterations=1)

        # Etiquetado de componentes y elección del más grande
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if num_labels <= 1:
            # Si no hay objeto, devolvemos todo cero
            return np.zeros_like(m)

        # stats[1:, CC_STAT_AREA] son las áreas de cada componente (excluye fondo)
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_id = 1 + np.argmax(areas)  # sumamos 1 porque saltamos el fondo
        return np.where(labels == max_id, 255, 0).astype(np.uint8)


# -----------------------------------------------------------------------------
# 4) Recorte cuadrado + normalización a OUTPUT_SIZE
# -----------------------------------------------------------------------------
class Cropper:
    def __init__(self, output_size: int = OUTPUT_SIZE, padding_ratio: float = 0.05):
        """
        output_size: lado del cuadrado final (ej. 256).
        padding_ratio: porcentaje de padding extra alrededor del bbox.
        """
        self.output = output_size
        self.pad_r = padding_ratio

    def crop_square(self, bgr: np.ndarray, mask: np.ndarray):
        """
        Recorta el bounding box del objeto (según mask) con un pequeño padding,
        lo reescala preservando aspecto y lo pega centrado en un lienzo cuadrado de output_size.
        Devuelve: (imagen_cuadrada, mascara_cuadrada)
        """
        ys, xs = np.where(mask > 0)
        if ys.size == 0:
            # Si no hay píxeles del objeto, devolvemos un lienzo negro
            s = self.output
            return np.zeros((s, s, 3), np.uint8), np.zeros((s, s), np.uint8)

        # Bounding box del objeto
        y1, y2 = ys.min(), ys.max()
        x1, x2 = xs.min(), xs.max()

        # Padding proporcional al tamaño del bbox
        h, w = y2 - y1 + 1, x2 - x1 + 1
        pad_y = int(self.pad_r * h); pad_x = int(self.pad_r * w)

        # Expandimos bbox con padding y recortamos cuidando límites de la imagen
        y1 = max(0, y1 - pad_y); y2 = min(bgr.shape[0] - 1, y2 + pad_y)
        x1 = max(0, x1 - pad_x); x2 = min(bgr.shape[1] - 1, x2 + pad_x)

        crop   = bgr[y1:y2+1, x1:x2+1]
        crop_m = mask[y1:y2+1, x1:x2+1]

        # Redimensionar a lienzo cuadrado manteniendo aspecto
        H, W = crop.shape[:2]; s = self.output
        scale = min(s / H, s / W)
        newH, newW = max(1, int(round(H * scale))), max(1, int(round(W * scale)))

        r_img = cv2.resize(crop,   (newW, newH), interpolation=cv2.INTER_AREA)
        r_msk = cv2.resize(crop_m, (newW, newH), interpolation=cv2.INTER_NEAREST)

        # Pegar centrado en lienzo s x s
        canvas   = np.zeros((s, s, 3), np.uint8)
        canvas_m = np.zeros((s, s), np.uint8)
        off_y = (s - newH) // 2
        off_x = (s - newW) // 2
        canvas[off_y:off_y+newH, off_x:off_x+newW] = r_img
        canvas_m[off_y:off_y+newH, off_x:off_x+newW] = r_msk
        return canvas, canvas_m


# -----------------------------------------------------------------------------
# 5) Extracción de características (HSV + HistH + HistV + forma)
# -----------------------------------------------------------------------------
class FeatureExtractor:

    def __init__(self, hue_bins: int = HUE_BINS, v_bins: int = V_BINS):
        """
        hue_bins: cantidad de bins para histograma de H (0..180 en OpenCV).
        v_bins:   cantidad de bins para histograma del valor (V).
        """
        self.hue_bins = int(hue_bins)
        self.v_bins   = int(v_bins)

        # Si alguna vez quisieras "borrar" bins de H (por varianza 0), ponelos aquí (1-based).
        # Con HUE_BINS=4 no vamos a eliminar nada.
        self.drop_histH_1based: List[int] = []

    # --- helpers de nombres (útil si querés exportar DataFrame con columnas) ---
    def _keep_histH_mask(self) -> np.ndarray:
        """
        Arma una máscara booleana de bins a conservar en histH.
        Por defecto son todos True (si drop_histH_1based está vacío).
        """
        keep = np.ones(self.hue_bins, dtype=bool)
        for b in self.drop_histH_1based:
            if 1 <= b <= self.hue_bins:
                keep[b - 1] = False
        return keep

    def get_feature_names(self) -> List[str]:
        """
        Nombres de features en el orden exacto que produce extract().
        """
        histH_names = [f"histH_{i}" for i in range(1, self.hue_bins + 1)]
        keep = self._keep_histH_mask()
        histH_names = [n for n, k in zip(histH_names, keep) if k]

        histV_names = [f"histV_{i}" for i in range(1, self.v_bins + 1)]

        return (["mean_H", "std_H", "mean_S", "std_S", "mean_V", "std_V"]  # 6 de color
                + histH_names                                            # histH filtrado
                + histV_names                                            # histV completo
                + ["circularity", "aspect_ratio", "extent",
                   "solidity", "area_rel", "perim_rel"])                 # 6 de forma

    def extract(self, bgr_sq: np.ndarray, mask_sq: np.ndarray) -> np.ndarray:
        """
        Extrae un vector de características del recorte cuadrado:
          - 6 stats de color (medias/std de H, S, V) EN LA REGIÓN DEL OBJETO
          - histograma normalizado de H (filtrado por _keep_histH_mask)
          - histograma normalizado de V
          - 6 features de forma: circularidad, aspecto, extent, solidez, area_rel, perim_rel
        """
        # Convertimos a HSV (OpenCV usa H en [0..180])
        hsv = cv2.cvtColor(bgr_sq, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

        # Máscara binaria 0/1
        m = (mask_sq > 0).astype(np.uint8)

        # --- 6 features de color dentro del objeto ---
        def _mean(x): return float(x.mean()) if x.size else 0.0
        def _std(x):  return float(x.std())  if x.size else 0.0
        h_m, s_m, v_m = h[m > 0], s[m > 0], v[m > 0]
        color_feats = np.array([_mean(h_m), _std(h_m),
                                _mean(s_m), _std(s_m),
                                _mean(v_m), _std(v_m)], dtype=np.float32)

        # --- Histograma de H (0..180), con máscara del objeto ---
        histH = cv2.calcHist([h], [0], m, [self.hue_bins], [0, 180]).flatten().astype(np.float32)
        histH = histH / (histH.sum() + 1e-6)  # normalizar
        histH = histH[self._keep_histH_mask()]  # opcionalmente eliminar bins

        # --- Histograma de V (0..255), con máscara del objeto ---
        histV = cv2.calcHist([v], [0], m, [self.v_bins], [0, 256]).flatten().astype(np.float32)
        histV = histV / (histV.sum() + 1e-6)  # normalizar

        # --- 6 features de forma a partir del contorno mayor ---
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            area = cv2.contourArea(c)
            per  = cv2.arcLength(c, True)

            # circularidad: 4π·area / per^2 (1 = círculo perfecto)
            circ = (4.0 * np.pi * area / (per * per)) if per > 0 else 0.0

            # bounding box axis-aligned (rectángulo no rotado)
            x, y, w, hbb = cv2.boundingRect(c)
            aspect = (w / hbb) if hbb > 0 else 0.0
            extent = (area / (w * hbb)) if (w * hbb) > 0 else 0.0

            # solidez = área / área del casco convexo
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            solidity = (area / hull_area) if hull_area > 0 else 0.0

            # relativas al tamaño del lienzo cuadrado
            Hc, Wc = m.shape[:2]
            area_rel  = area / float(Hc * Wc)
            perim_rel = per  / float(Hc + Wc)

            shape_feats = np.array([circ, aspect, extent, solidity, area_rel, perim_rel],
                                   dtype=np.float32)
        else:
            # Sin contornos -> cero
            shape_feats = np.zeros(6, dtype=np.float32)

        # Concatenamos todo en un único vector 1D (float32)
        return np.concatenate([color_feats, histH, histV, shape_feats]).astype(np.float32)


# -----------------------------------------------------------------------------
# 6) Visualización auxiliar: overlay de contorno + bbox (para debug)
# -----------------------------------------------------------------------------
class Visualizer:
    def overlay(self, bgr_sq: np.ndarray, mask_sq: np.ndarray) -> np.ndarray:
        """
        Dibuja el contorno y la caja del objeto sobre la imagen cuadrada (para inspección).
        """
        vis = bgr_sq.copy()
        m = (mask_sq > 0).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            cv2.drawContours(vis, [c], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return vis


# -----------------------------------------------------------------------------
# 7) Orquestador del pipeline (usa las clases anteriores)
# -----------------------------------------------------------------------------
class ProcessingPipeline:
    #constructor
    def __init__(self, output_size: int = OUTPUT_SIZE,
                 hue_bins: int = HUE_BINS, v_bins: int = V_BINS):
        # Componentes del pipeline
        self.segmenter  = Segmenter()                       # segmentación del objeto
        self.cropper    = Cropper(output_size=output_size)  # recorte + normalización a cuadrado
        self.extractor  = FeatureExtractor(hue_bins=hue_bins, v_bins=v_bins)  # features
        self.visualizer = Visualizer()                      # debug/overlay

        # Guardamos los parámetros usados (por claridad y futura inspección)
        self.output_size = output_size
        self.hue_bins    = hue_bins
        self.v_bins      = v_bins

    def get_feature_names(self) -> List[str]:
        return self.extractor.get_feature_names()

    def process_one(self, file_path: str, dest_dir: str, label: str):
        """
        Procesa una sola imagen:
          - lee + segmenta + recorta a cuadrado
          - extrae features
          - guarda (original/mask/overlay) en dest_dir/label/...
          - devuelve (vector_de_features, label)
        """
        bgr = safe_imread(file_path)
        if bgr is None:
            print(f"[WARN] No se pudo leer: {file_path}")
            return None, None

        mask = self.segmenter.segment(bgr)
        bgr_sq, m_sq = self.cropper.crop_square(bgr, mask)
        feats = self.extractor.extract(bgr_sq, m_sq)

        # Guardar imágenes procesadas para inspección
        os.makedirs(os.path.join(dest_dir, label), exist_ok=True)
        base = os.path.splitext(os.path.basename(file_path))[0]
        out_dir = os.path.join(dest_dir, label)
        cv2.imwrite(os.path.join(out_dir, f"{base}_original.png"), bgr_sq)
        cv2.imwrite(os.path.join(out_dir, f"{base}_mask.png"),     m_sq)
        cv2.imwrite(os.path.join(out_dir, f"{base}_overlay.png"),  self.visualizer.overlay(bgr_sq, m_sq))

        return feats, label

    def process_directory(self, source_dir: str, dest_dir: str, mode: str = "flat"):
        """
        Procesa todas las imágenes en 'source_dir' (forma plana o subcarpetas).
        Devuelve:
          - X: matriz de features (N, D)
          - y: etiquetas (N,)
          - paths: rutas originales (lista de N)
        """
        pairs = list_images(source_dir)  # [(label, path), ...]
        X_list, y_list, p_list = [], [], []

        for label, path in pairs:
            feats, lab = self.process_one(path, dest_dir, label)
            if feats is None:
                continue
            X_list.append(feats)
            y_list.append(lab)
            p_list.append(path)

        if not X_list:
            # Sin datos -> devolvemos shapes coherentes
            return np.zeros((0, len(self.get_feature_names())), dtype=np.float32), np.array([]), []

        X = np.stack(X_list).astype(np.float32)  # (N, D)
        y = np.array(y_list)                     # (N,)
        return X, y, p_list


# -----------------------------------------------------------------------------
# 8) Z-score persistible y mapeo cluster->etiqueta (por voto mayoritario)
# -----------------------------------------------------------------------------
class Standardizer:
    """ Z-score con persistencia (guarda/carga mu y sigma). """
    def __init__(self, mu=None, sigma=None):
        self.mu = mu
        self.sigma = sigma

    def fit(self, X):
        mu = X.mean(axis=0, keepdims=True)
        sigma = X.std(axis=0, keepdims=True)
        sigma[sigma == 0] = 1.0  # evitar división por 0
        self.mu, self.sigma = mu, sigma
        return self

    def transform(self, X):
        return (X - self.mu) / self.sigma

    def save(self, dirpath):
        os.makedirs(dirpath, exist_ok=True)
        np.save(os.path.join(dirpath, "mu.npy"),    self.mu)
        np.save(os.path.join(dirpath, "sigma.npy"), self.sigma)

    @classmethod
    def load(cls, dirpath):
        mu    = np.load(os.path.join(dirpath, "mu.npy"))
        sigma = np.load(os.path.join(dirpath, "sigma.npy"))
        return cls(mu, sigma)



"""
    Mapea id de cluster -> etiqueta humana por voto mayoritario
    usando las etiquetas verdaderas durante entrenamiento.
"""
class ClusterLabelMapper:
    
    def __init__(self, mapping=None):
        self.mapping = mapping or {}

    def fit(self, y_true, clusters):
        mapping = {}
        for k in np.unique(clusters):
            idx = (clusters == k)
            if idx.sum() == 0:
                continue
            # etiqueta más frecuente dentro del clúster k
            mapping[int(k)] = Counter(y_true[idx]).most_common(1)[0][0]
        self.mapping = mapping
        return self

    def transform(self, clusters):
        # Devuelve la etiqueta humana para cada id de clúster
        return np.array([self.mapping.get(int(c), f"cluster_{int(c)}") for c in clusters])

    def save(self, dirpath):
        with open(os.path.join(dirpath, "cluster_to_label.json"), "w", encoding="utf-8") as f:
            json.dump(self.mapping, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, dirpath):
        with open(os.path.join(dirpath, "cluster_to_label.json"), "r", encoding="utf-8") as f:
            m = json.load(f)
        # claves a int por las dudas
        return cls({int(k): v for k, v in m.items()})


# -----------------------------------------------------------------------------
# 9) Helpers de inicialización "seeded" (promedio por clase)
# -----------------------------------------------------------------------------
def _class_means(Xz: np.ndarray, y: np.ndarray):
    """
    Calcula centroides iniciales deterministas como la MEDIA de cada clase.
    Devuelve: (matriz_centroides (k,D), clases_en_orden).
    """
    classes = np.unique(y)
    #Xz matriz de características estandarizadas
    #Y vector de etiquetas
    #cents es la lista de centroides por clase
    cents = [Xz[y == c].mean(axis=0) for c in classes]
    return np.vstack(cents), classes #devuelve una tupla matriz de centroides 


# -----------------------------------------------------------------------------
# 10) Entrenamiento desde carpeta (procesa -> zscore -> KMeans -> guarda artefactos)
# -----------------------------------------------------------------------------
def _confusion_matrix(y_true, y_pred):
    classes = sorted(list(set(y_true) | set(y_pred)))
    idx = {c: i for i, c in enumerate(classes)}
    M = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true, y_pred):
        M[idx[t], idx[p]] += 1
    return classes, M


def train_from_directory(source_dir: str,
                         dest_dir: str,
                         artifact_dir: str,
                         mode: str = "flat",
                         k: int = 4,                    # tus 4 clases (berenjena, camote, papa, zanahoria)
                         random_state: int = 42,
                         init_strategy: str = "seeded", # 'seeded' | 'kmeans++' | 'random'
                         max_iter: int = 100,
                         tol: float = 1e-6):
    """
    Procesa la carpeta, estandariza (Z-score), entrena K-Means y guarda artefactos:
      - mu.npy / sigma.npy    (Z-score)
      - centroids.npy         (centroides K-Means)
      - cluster_to_label.json (mapeo cluster -> etiqueta humana)
    También imprime accuracy y matriz de confusión en entrenamiento.
    """
    os.makedirs(dest_dir, exist_ok=True)
    os.makedirs(artifact_dir, exist_ok=True)

    # 1) Procesar dataset completo -> X (features), y (etiquetas), paths
    pipe = ProcessingPipeline(output_size=OUTPUT_SIZE, hue_bins=HUE_BINS, v_bins=V_BINS)
    X, y, paths = pipe.process_directory(source_dir, dest_dir, mode=mode)
    print(f"Imágenes procesadas: {len(paths)}")
    if X.shape[0] == 0:
        print("[WARN] No hay features para entrenar.")
        return 0.0, np.array([]), [], None

    # 2) Calcular Z-score y transformar
    std = Standardizer().fit(X)
    Xz  = std.transform(X)

    # 3) Inicialización 'seeded' (si fue seleccionada)
    init_cents = None
    if init_strategy == "seeded":
        # centroides iniciales = media por clase (determinista)
        init_cents, _ = _class_means(Xz, y)

        # Asegurar que la cantidad de centroides coincide con k (por si cambiaste k)
        if init_cents.shape[0] != k:
            if init_cents.shape[0] > k:
                # Si hay más centroides que k, recortamos (poco habitual)
                init_cents = init_cents[:k]
            else:
                # Si faltan centroides, completamos con filas reales elegidas de forma reproducible
                rng = np.random.RandomState(random_state)
                extra_idx = rng.choice(len(Xz), size=(k - init_cents.shape[0]), replace=False)
                init_cents = np.vstack([init_cents, Xz[extra_idx]])

    # 4) Entrenar K-Means (tu implementación)
    km = KMeans(n_clusters=k,
                init=init_strategy,             # 'seeded'|'kmeans++'|'random' (la lógica interna está en kmeans.py)
                random_state=random_state,
                max_iter=max_iter,
                tol=tol)

    # Si init='seeded', pasamos los centroides iniciales calculados arriba.
    labels = km.fit(Xz, init_centroids=init_cents)
    centroids = km.centroids

    # 5) Mapeo cluster->etiqueta (voto mayoritario)
    mapper = ClusterLabelMapper().fit(y_true=y, clusters=labels)
    y_pred = mapper.transform(labels)

    # 6) Métricas de entrenamiento
    acc = (y_pred == y).mean()
    print(f"Acierto (voto mayoritario): {acc*100:.1f}%")

    classes, M = _confusion_matrix(y, y_pred)
    print("Matriz de confusión (filas=verdadero, columnas=predicho):")
    # Encabezado de columnas
    print("      " + "  ".join([f"{c:>10.10}" for c in classes]))
    # Filas
    for i, c in enumerate(classes):
        print(f"{c:>5.5} " + " ".join([f"{v:10d}" for v in M[i]]))

    # 7) Guardar artefactos para uso posterior (clasificación de nuevas imágenes)
    std.save(artifact_dir)  # mu.npy, sigma.npy
    np.save(os.path.join(artifact_dir, "centroids.npy"), centroids)
    mapper.save(artifact_dir)  # cluster_to_label.json

    # Devolver algunos resultados útiles para scripts externos (train_main.py, run_menu.py, etc.)
    return acc, np.unique(y), paths, (std, centroids, mapper)


# -----------------------------------------------------------------------------
# 11) Clasificar una imagen nueva usando artefactos guardados
# -----------------------------------------------------------------------------
def classify_new_image(image_path: str, artifact_dir: str, dest_dir: str):
    """
    Clasifica una imagen nueva:
      - Carga mu/sigma + centroides + mapping
      - Procesa la imagen NUEVA con el MISMO pipeline (mismos HUE_BINS/V_BINS/OUTPUT_SIZE)
      - Estandariza con mu/sigma entrenados
      - Asigna al centroide más cercano
      - Devuelve la etiqueta humana (según mapping)
    """
    # 1) Cargar artefactos
    std      = Standardizer.load(artifact_dir)
    centroids = np.load(os.path.join(artifact_dir, "centroids.npy"))
    mapper   = ClusterLabelMapper.load(artifact_dir)

    # 2) Procesar la imagen (se guarda en dest_dir/desconocido para inspección)
    pipe = ProcessingPipeline(output_size=OUTPUT_SIZE, hue_bins=HUE_BINS, v_bins=V_BINS)
    X_new, _ = pipe.process_one(image_path, dest_dir, label="desconocido")
    if X_new is None:
        print("[WARN] No se pudo extraer features.")
        return None

    # 3) Z-score con parámetros del entrenamiento
    Xz_new = std.transform(X_new.reshape(1, -1))

    # 4) Asignación por centroide más cercano (sin reentrenar)
    d = np.linalg.norm(Xz_new[:, None, :] - centroids[None, :, :], axis=2)  # (1,k)
    c = int(d.argmin(axis=1)[0])  # id de clúster

    # 5) Mapear a etiqueta humana y devolver
    pred = mapper.transform(np.array([c]))[0]
    print(f"Imagen '{os.path.basename(image_path)}' → '{pred}' (cluster {c})")
    return pred
