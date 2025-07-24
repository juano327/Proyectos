import numpy as np
import matplotlib.pyplot as plt

# Función para generar un conjunto de 23 puntos aleatorios en el intervalo [0, 5]
def generar_puntos_aleatorios(num_puntos=23, rango=(0, 5)):
    return np.random.uniform(rango[0], rango[1], (num_puntos, 2))

# Función para asignar clusters a los puntos basándose en la distancia euclidiana a los centroides
def asignar_clusters(datos, centroides):
    clusters = []
    for punto in datos:
        distancias = [np.linalg.norm(punto - centroide) for centroide in centroides]
        clusters.append(np.argmin(distancias))
    return clusters

# Función para recalcular los centroides
def recalcular_centroides(datos, clusters, k):
    nuevos_centroides = []
    for i in range(k):
        puntos_cluster = datos[np.array(clusters) == i]
        if len(puntos_cluster) > 0:
            nuevos_centroides.append(np.mean(puntos_cluster, axis=0))
        else:
            nuevos_centroides.append(datos[np.random.choice(len(datos))])
    return np.array(nuevos_centroides)

# Función para verificar la convergencia
def ha_convergido(centroides_anteriores, centroides_nuevos, tolerancia=1e-4):
    return np.linalg.norm(centroides_nuevos - centroides_anteriores) < tolerancia

# Implementación del algoritmo K-means
def k_means(datos, k=2, max_iter=100):
    centroides = datos[np.random.choice(len(datos), k, replace=False)]
    for _ in range(max_iter):
        clusters = asignar_clusters(datos, centroides)
        nuevos_centroides = recalcular_centroides(datos, clusters, k)
        if ha_convergido(centroides, nuevos_centroides):
            break
        centroides = nuevos_centroides
    return clusters, centroides

# Función KNN para clasificar un punto nuevo
def knn_clasificar(punto, datos, clusters, k=3):
    distancias = [np.linalg.norm(punto - d) for d in datos]
    vecinos_indices = np.argsort(distancias)[:k]
    vecinos_clusters = [clusters[i] for i in vecinos_indices]
    return max(set(vecinos_clusters), key=vecinos_clusters.count)

# 3: Generar los puntos aleatorios y dividirlos en conjunto de entrenamiento y prueba
puntos = generar_puntos_aleatorios()
entrenamiento = puntos[:20]
prueba = puntos[20:]

# Paso 3.1: Aplicar K-means al conjunto de entrenamiento
clusters, centroides = k_means(entrenamiento, k=2)

# Graficar los resultados del K-means
plt.figure(figsize=(8, 6))
for i in range(2):
    puntos_cluster = entrenamiento[np.array(clusters) == i]
    plt.scatter(puntos_cluster[:, 0], puntos_cluster[:, 1], label=f'Cluster {i+1}')
plt.scatter(centroides[:, 0], centroides[:, 1], c='red', marker='x', s=100, label='Centroides')
plt.title('Resultados de K-means')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

# Paso 3.2: Clasificar los puntos de prueba con KNN y probar con diferentes valores de K
for k in [1, 3, 5]:
    print(f"\nClasificación usando KNN con K={k}:")
    for punto in prueba:
        cluster_predicho = knn_clasificar(punto, entrenamiento, clusters, k=k)
        print(f"Punto {punto} asignado al cluster {cluster_predicho}")
