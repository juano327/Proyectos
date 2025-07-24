import pygame
import sys
import time

# Inicializar pygame
pygame.init()

# Configuración de la pantalla
ANCHO, ALTO = 600, 400
COLOR_FONDO = (0, 0, 0)
COLOR_RELLENO = (200, 200, 200)
COLOR_CAMINO = (0, 255, 0)
COLOR_NODO_ACTUAL = (255, 0, 0)
TAM_CELDA = 100
pantalla = pygame.display.set_mode((ANCHO, ALTO))
pygame.display.set_caption('Visualización de Búsqueda Primero en Profundidad')

def dibujar_camino(camino):
    for nodo in camino:
        fila, col = convertir_a_coordenadas(nodo.valor)
        pygame.draw.rect(pantalla, COLOR_CAMINO, pygame.Rect(col * TAM_CELDA, fila * TAM_CELDA, TAM_CELDA, TAM_CELDA))

def dibujar_nodo_actual(nodo):
    fila, col = convertir_a_coordenadas(nodo.valor)
    pygame.draw.rect(pantalla, COLOR_NODO_ACTUAL, pygame.Rect(col * TAM_CELDA, fila * TAM_CELDA, TAM_CELDA, TAM_CELDA))

def convertir_a_coordenadas(valor):
    # Convierte el valor del nodo a coordenadas en la matriz
    coordenadas = {
        'I': (2, 1), 'G': (2, 0), 'P': (3, 0), 'Q': (3, 1), 'R': (3, 2), 'T': (3, 3),
        'W': (2, 2), 'K': (2, 3), 'M': (2, 4), 'N': (2, 5), 'E': (1, 5), 'D': (1, 4),
        'C': (1, 3), 'B': (0, 4), 'A': (0, 3), 'F': (3, 4)
    }
    return coordenadas[valor]

# clase Nodo
class Nodo:
    def __init__(self, valor):
        self.valor = valor
        self.hijos = []

def busqueda_profundidad(nodo_inicio, valor_objetivo, pantalla):
    pila = [nodo_inicio]
    visitados = set()
    camino = []

    while pila:
        nodo_actual = pila.pop(0)
        camino.append(nodo_actual)

        # Actualizar la pantalla
        pantalla.fill(COLOR_FONDO)
        dibujar_matriz()
        dibujar_camino(camino)
        dibujar_nodo_actual(nodo_actual)
        pygame.display.flip()
        time.sleep(1)  # Pausa para mostrar el estado actual

        print(f"Visitando nodo: {nodo_actual.valor}")

        if nodo_actual.valor == valor_objetivo:
            print("¡Objetivo encontrado!")
            return camino

        visitados.add(nodo_actual)

        for hijo in reversed(nodo_actual.hijos):
            if hijo not in visitados:
                pila.insert(0, hijo)

    print("Objetivo no encontrado.")
    return None

def dibujar_matriz():
    matriz = [
        ['X', 'X', 'X', 'A', 'B', 'X'],
        ['X', 'X', 'X', 'C', 'D', 'E'],
        ['G', 'I', 'W', 'K', 'M', 'N'],
        ['P', 'Q', 'R', 'T', 'F', 'X']
    ]
    for fila in range(len(matriz)):
        for col in range(len(matriz[fila])):
            color = COLOR_RELLENO if matriz[fila][col] != 'X' else COLOR_FONDO
            pygame.draw.rect(pantalla, color, pygame.Rect(col * TAM_CELDA, fila * TAM_CELDA, TAM_CELDA, TAM_CELDA))
            pygame.draw.rect(pantalla, (0, 0, 0), pygame.Rect(col * TAM_CELDA, fila * TAM_CELDA, TAM_CELDA, TAM_CELDA), 1)
            if matriz[fila][col] != 'X':
                fuente = pygame.font.SysFont(None, 30)
                texto = fuente.render(matriz[fila][col], True, (0, 0, 0))
                pantalla.blit(texto, (col * TAM_CELDA + 10, fila * TAM_CELDA + 10))

# crear el grafo
grafo = {
    'I': ['G', 'Q', 'W'],
    'G': ['I', 'P'],
    'P': ['G', 'Q'],
    'Q': ['I', 'P', 'R'],
    'R': ['Q', 'T'],
    'T': ['K', 'R', 'F'],
    'W': ['I', 'K'],
    'K': ['C', 'M', 'T', 'W'],
    'M': ['D', 'F', 'N', 'K'],
    'N': ['E', 'M'],
    'E': ['N'],
    'D': ['B', 'M'],
    'C': ['A', 'K'],
    'B': ['A', 'D'],
    'A': ['B', 'C'],
    'F': ['M', 'T']
}

# crear los nodos del grafo
nodos = {}
for clave in grafo:
    nodos[clave] = Nodo(clave)

# conectar los nodos del grafo
for clave, valor in grafo.items():
    nodo = nodos[clave]
    hijos = [nodos[hijo] for hijo in valor]
    nodo.hijos = hijos

# realizar la búsqueda en profundidad
inicio = nodos['I']
valor_objetivo = 'F'
camino = busqueda_profundidad(inicio, valor_objetivo, pantalla)

if camino:
    print("\nCamino recorrido:")
    for nodo in camino:
        print(nodo.valor, end=" -> ")
    print("FIN")
else:
    print("No se encontró un camino al objetivo.")

# Mantener la ventana abierta hasta que el usuario la cierre
ejecutando = True
while ejecutando:
    for evento in pygame.event.get():
        if evento.type == pygame.QUIT:
            ejecutando = False

pygame.quit()
