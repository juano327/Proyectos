import numpy as np
import random as rd

class Individuo:
    def __init__(self, elementos):
        # Elementos contiene una lista de diccionarios con el formato {'precio': int, 'peso': int}
        self.elementos = elementos.copy()  # Se usa copia para evitar mutaciones no deseadas en los cruces
        self.precio = 0
        self.peso = 0
        self.cargar()
    
    def cargar(self):
        """Carga los elementos del individuo y calcula el precio total y el peso total."""
        self.precio = 0
        self.peso = 0
        for i in range(len(self.elementos)):
            if rd.random() > 0.5:  # Decidir si el elemento se toma o no
                self.precio += self.elementos[i]["precio"]
                self.peso += self.elementos[i]["peso"]

def evaluar_poblacion(poblacion, C):
    """Evalúa la población, eliminando el valor de los individuos que excedan el límite de peso."""
    for individuo in poblacion:
        if individuo.peso > C:
            individuo.precio = 0

def seleccionar_padres(poblacion):
    """Selecciona N individuos como padres basándose en la ruleta proporcional a su precio."""
    padres = []
    precios = np.array([individuo.precio for individuo in poblacion])
    if np.sum(precios) == 0:
        return []
    probabilidades = precios / np.sum(precios)
    for _ in range(len(poblacion)//2):  # Seleccionamos N/2 parejas
        padres.append(rd.choices(poblacion, probabilidades)[0])
    return padres

def cruzar(padres):
    """Cruza los padres para generar una nueva población de hijos."""
    hijos = []
    for padre1, padre2 in zip(padres[::2], padres[1::2]):
        hijo1 = Individuo(padre1.elementos)
        hijo2 = Individuo(padre2.elementos)
        for i in range(len(hijo1.elementos)):
            if rd.random() > 0.5:  # Intercambiamos elementos entre los padres
                hijo1.elementos[i] = padre2.elementos[i]
                hijo2.elementos[i] = padre1.elementos[i]
        hijo1.cargar()
        hijo2.cargar()
        hijos.append(hijo1)
        hijos.append(hijo2)
    return hijos

def mutar(poblacion):
    """Aplica mutación a la población con una baja probabilidad."""
    for individuo in poblacion:
        if rd.random() < 0.1:  # Probabilidad de mutación del 10%
            i = rd.randint(0, len(individuo.elementos)-1)  # Seleccionamos un elemento aleatorio
            individuo.elementos[i] = {"precio": rd.randint(10, 200), "peso": rd.randint(50, 500)}  # Mutamos el elemento
            individuo.cargar()  # Recalculamos el precio y el peso del individuo

def mejor_individuo(poblacion):
    """Retorna el mejor individuo de la población en términos de precio."""
    if not poblacion:
        return None
    
    precios = np.array([individuo.precio for individuo in poblacion])
    
    if np.all(precios == 0):
        return None
    
    return poblacion[np.argmax(precios)]

# Datos de las cajas (elementos)
elementos = [
    {"precio": 100, "peso": 300},
    {"precio": 50, "peso": 200},
    {"precio": 115, "peso": 450},
    {"precio": 25, "peso": 145},
    {"precio": 200, "peso": 664},
    {"precio": 30, "peso": 90},
    {"precio": 40, "peso": 150},
    {"precio": 100, "peso": 355},
    {"precio": 100, "peso": 401},
    {"precio": 100, "peso": 395}
]

# Parámetros
N = 100
C = 1000

# Inicialización de la población
poblacion = [Individuo(elementos) for i in range(N)]
evaluar_poblacion(poblacion, C)

# Buscar el mejor individuo inicial
mejor = mejor_individuo(poblacion)
if mejor is None:
    print("No hay individuos válidos en la población inicial.")
else:
    print(f"Mejor inicial - Precio: {mejor.precio}, Peso: {mejor.peso}")

# Bucle evolutivo (100 iteraciones)
for i in range(100):
    padres = seleccionar_padres(poblacion)
    if not padres:
        print("No se pudieron seleccionar padres.")
        break

    hijos = cruzar(padres)
    mutar(hijos)
    evaluar_poblacion(hijos, C)

    # Mantener la población del mismo tamaño
    poblacion = hijos + padres[:len(hijos)]  # Mezclamos parte de los padres con los hijos si es necesario
    mejor = mejor_individuo(poblacion)
        
    if mejor is None:
        print("No hay individuos mejores en la población.")
        break

    print(f"Iteración {i+1} - Mejor Precio: {mejor.precio}, Peso: {mejor.peso}")

# Imprimir el mejor individuo final
mejor_final = mejor_individuo(poblacion)
if mejor_final:
    print(f"Mejor final - Precio: {mejor_final.precio}, Peso: {mejor_final.peso}")
else:
    print("No se encontró un mejor individuo.")
