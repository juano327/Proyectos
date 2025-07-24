import numpy as np
import random
import tkinter as tk
from tkinter import messagebox

class TaTeTi:
    def __init__(self):
        # inicializamos el tablero vacío (0: vacío, 1: jugador, -1: IA)
        self.tablero = np.zeros((3, 3), dtype=int)

    def realizar_movimiento(self, fila, columna, jugador):
        # si la casilla está vacía, se realiza el movimiento
        if self.tablero[fila, columna] == 0:
            self.tablero[fila, columna] = jugador
            return True
        return False

    def verificar_ganador(self):
        # comprobamos si hay un ganador en filas, columnas o diagonales
        for i in range(3):
            if np.all(self.tablero[i, :] == 1) or np.all(self.tablero[:, i] == 1):
                return 1  # gana el jugador
            if np.all(self.tablero[i, :] == -1) or np.all(self.tablero[:, i] == -1):
                return -1  # gana la IA
        if np.all(np.diag(self.tablero) == 1) or np.all(np.diag(np.fliplr(self.tablero)) == 1):
            return 1
        if np.all(np.diag(self.tablero) == -1) or np.all(np.diag(np.fliplr(self.tablero)) == -1):
            return -1
        if np.all(self.tablero != 0):
            return 0  # empate
        return None  # el juego continúa

    def reiniciar(self):
        # reiniciamos el tablero para una nueva partida
        self.tablero = np.zeros((3, 3), dtype=int)

class RecocidoSimuladoIA:
    def __init__(self, temperatura_inicial):
        # inicializamos la temperatura del algoritmo
        self.temperatura = temperatura_inicial

    def obtener_movimientos_validos(self, tablero):
        # devolvemos una lista de posiciones vacías (movimientos válidos)
        return [(fila, col) for fila in range(3) for col in range(3) if tablero[fila, col] == 0]

    def evaluar_tablero(self, tablero):
        # evaluamos el estado del tablero para saber si hay un ganador
        ganador = TaTeTi().verificar_ganador()
        if ganador == -1:
            return 1  # gana la IA
        elif ganador == 1:
            return -1  # gana el jugador
        return 0  # empate o el juego continúa

    def movimiento_recocido_simulado(self, juego):
        # obtenemos los movimientos válidos
        movimientos_validos = self.obtener_movimientos_validos(juego.tablero)
        mejor_movimiento = None
        mejor_puntaje = float('-inf')

        # probamos todos los movimientos posibles
        for movimiento in movimientos_validos:
            juego.tablero[movimiento] = -1  # simulamos un movimiento de la IA
            puntaje = self.evaluar_tablero(juego.tablero)
            # aplicamos la lógica del recocido simulado para elegir el mejor movimiento
            if puntaje > mejor_puntaje or random.uniform(0, 1) < np.exp(-abs(puntaje - mejor_puntaje) / self.temperatura):
                mejor_movimiento = movimiento
                mejor_puntaje = puntaje
            juego.tablero[movimiento] = 0  # deshacemos el movimiento simulado

        # reducimos la temperatura en cada iteración
        self.temperatura *= 0.95
        return mejor_movimiento


class InterfazJuego:
    def __init__(self, raiz):
        # inicializamos la ventana principal e instancias del juego
        self.raiz = raiz
        self.raiz.title("TA-TE-TI")
        self.juego = TaTeTi()
        self.ia = RecocidoSimuladoIA(temperatura_inicial=1.0)  # dificultad predeterminada
        self.botones = [[None for _ in range(3)] for _ in range(3)]
        self.dificultad = tk.DoubleVar(value=1.0)
        self.crear_interfaz()

    def crear_interfaz(self):
        # creamos los widgets de la interfaz gráfica
        tk.Label(self.raiz, text="Seleccionar Dificultad:").grid(row=0, column=0, columnspan=3)
        tk.Scale(self.raiz, from_=0.1, to=2.0, resolution=0.1, variable=self.dificultad,
                 orient=tk.HORIZONTAL).grid(row=1, column=0, columnspan=3)
        # creación de los botones del tablero
        for fila in range(3):
            for col in range(3):
                self.botones[fila][col] = tk.Button(self.raiz, text="", font=('normal', 40), width=5, height=2,
                                                    command=lambda fila=fila, col=col: self.movimiento_jugador(fila, col))
                self.botones[fila][col].grid(row=fila + 2, column=col)

        # botón para reiniciar el juego
        self.boton_reiniciar = tk.Button(self.raiz, text="Reiniciar", command=self.reiniciar_juego)
        self.boton_reiniciar.grid(row=5, column=0, columnspan=3)

    def movimiento_jugador(self, fila, col):
        # el jugador realiza un movimiento
        if self.juego.realizar_movimiento(fila, col, 1):
            self.actualizar_botones()
            ganador = self.juego.verificar_ganador()
            if ganador is not None:
                self.mostrar_resultado(ganador)
                return
            self.movimiento_ia()

    def movimiento_ia(self):
        # la IA realiza un movimiento utilizando recocido simulado
        self.ia.temperatura = self.dificultad.get()  # ajustamos la temperatura según la dificultad seleccionada
        movimiento = self.ia.movimiento_recocido_simulado(self.juego)
        if movimiento:
            self.juego.realizar_movimiento(movimiento[0], movimiento[1], -1)
        self.actualizar_botones()
        ganador = self.juego.verificar_ganador()
        if ganador is not None:
            self.mostrar_resultado(ganador)

    def actualizar_botones(self):
        # actualizamos la interfaz gráfica para reflejar el estado actual del tablero
        for fila in range(3):
            for col in range(3):
                texto = "X" if self.juego.tablero[fila, col] == 1 else "O" if self.juego.tablero[fila, col] == -1 else ""
                self.botones[fila][col].config(text=texto)

    def mostrar_resultado(self, ganador):
        # mostramos un mensaje con el resultado del juego
        if ganador == 1:
            messagebox.showinfo("Resultado", "¡Ganaste! :D")
        elif ganador == -1:
            messagebox.showinfo("Resultado", "Perdiste :(")
        else:
            messagebox.showinfo("Resultado", "¡Empate! :|")
        self.reiniciar_juego()

    def reiniciar_juego(self):
        # Reiniciamos el juego
        self.juego.reiniciar()
        self.actualizar_botones()

if __name__ == "__main__":
    raiz = tk.Tk()
    interfaz = InterfazJuego(raiz)
    raiz.mainloop()
