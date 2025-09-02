# kmeans.py
import numpy as np
class KMeans:
    """
    K-Means sin scikit-learn, con inicializaciones:
      - init='random'    : centroides al azar (reproducible con random_state)
      - init='kmeans++'  : K-Means++ determinista usando random_state
      - init='seeded'    : usa centroides iniciales provistos (init_centroids)

    API:
      fit(X, init_centroids=None) -> labels (np.ndarray de shape (n,))
      predict(X) -> labels usando self.centroids
      save(dirpath), load(dirpath)
    """
    def __init__(self, n_clusters=4, max_iter=100, tol=1e-6, random_state=42, init='random'):
        self.n_clusters = int(n_clusters)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = int(random_state)
        self.init = str(init)
        self.centroids = None

    # ---------- Inicializaciones ----------
    def _random_init(self, X):
        rng = np.random.RandomState(self.random_state)
        idx = rng.choice(len(X), size=self.n_clusters, replace=False)
        return X[idx].copy()

    def _kmeanspp_init(self, X):
        rng = np.random.RandomState(self.random_state)
        n = len(X)
        first = rng.randint(0, n)
        cents = [X[first]]
        for _ in range(1, self.n_clusters):
            # dist^2 mínima a un centroide ya elegido
            d2 = np.min(np.linalg.norm(X[:, None, :] - np.vstack(cents)[None, :, :], axis=2) ** 2, axis=1)
            probs = d2 / (d2.sum() + 1e-12)
            idx = rng.choice(n, p=probs)
            cents.append(X[idx])
        return np.vstack(cents)

    # ---------- Core ----------
    @staticmethod
    #“Para cada punto del dataset, calculo a qué centroide le queda más cerca (en distancia euclidiana) 
    # y me quedo con ese índice de clúster.”
    def _assign(X, centroids):
        #X tiene shape (N muestras, D características)
        # centroids tiene shape (K centroides, D características)
        d = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
            #calcula la norma euclidiana a lo largo de D 
            #produce (N, K, D) con todas las restas punto‑menos‑centroide.
        return d.argmin(axis=1)#obtienes una matriz de distancias d con shape (N, K) (distancia de cada punto a cada centroide).
        #devuelve, para cada una de las N filas, el índice k del centroide más cercano → vector (N,) con las asignaciones de clúster.
    def fit(self, X, init_centroids=None):
        X = np.asarray(X)
        if X.ndim != 2:
            X = X.reshape(len(X), -1)

        # elegir centroides iniciales
        if self.init == 'seeded':
            if init_centroids is None:
                raise ValueError("init='seeded' requiere init_centroids")
            centroids = np.asarray(init_centroids).copy()
            if centroids.shape != (self.n_clusters, X.shape[1]):
                raise ValueError(f"init_centroids debe tener shape ({self.n_clusters}, {X.shape[1]})")
        elif self.init == 'kmeans++':
            centroids = self._kmeanspp_init(X)
        else:  # 'random'
            centroids = self._random_init(X)

        last_inertia = None
        for _ in range(self.max_iter):
            labels = self._assign(X, centroids) #asigna cada punto a su cluster. es un vector(N,) con ids de clúster.
            new_centroids = []
            for k in range(self.n_clusters):
                pts = X[labels == k] #→ submatriz (n_k, D) con los puntos del clúster k.
                if len(pts) == 0:
                    # cluster vacío: dejar el centroide como estaba (determinista)
                    new_centroids.append(centroids[k])
                else:
                    new_centroids.append(pts.mean(axis=0)) #calcula la media de cada columna (feature) 
                                                #→ eso da un nuevo vector en ℝᴰ que será el centroide.
                                                #Si no está vacío, el nuevo centroide es la media por columnas:
            new_centroids = np.vstack(new_centroids)
            centroids = new_centroids #Se pisa el estado con los centroides recalculados

            inertia = ((X - centroids[labels]) ** 2).sum() #qué tan apretados quedaron los puntos respecto a sus centroides. error cuaratico
            if last_inertia is not None and abs(last_inertia - inertia) < self.tol:
                break
            last_inertia = inertia

        self.centroids = centroids
        return labels

    def predict(self, X):
        #con los centroides ya entrenados, asigno cada nueva muestra a su clúster más cercano
        if self.centroids is None:
            raise RuntimeError("KMeans no entrenado. Llamá fit() primero o cargá centroides.")
        X = np.asarray(X)
        if X.ndim != 2:
            X = X.reshape(len(X), -1)
        return self._assign(X, self.centroids)

    # ---------- Persistencia ----------
    def save(self, dirpath):
        import os
        os.makedirs(dirpath, exist_ok=True)
        np.save(os.path.join(dirpath, "centroids.npy"), self.centroids)

    @classmethod
    def load(cls, dirpath):
        import os
        cent = np.load(os.path.join(dirpath, "centroids.npy"))
        obj = cls(n_clusters=len(cent))
        obj.centroids = cent
        return obj
