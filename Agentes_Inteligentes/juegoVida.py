import sys
import pygame
import random

# definición de la clase para el Juego de la Vida
class JuegoDeLaVida:
    def __init__(self, width, height, cell_size):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.columns = width // cell_size
        self.rows = height // cell_size
        self.grid = self.crear_grid_aleatoria()
        
    def crear_grid_aleatoria(self):
        return [[random.choice([0, 1]) for _ in range(self.columns)] for _ in range(self.rows)]

    def actualizar_grid(self):
        nueva_grid = [[0 for _ in range(self.columns)] for _ in range(self.rows)]
        
        for y in range(self.rows):
            for x in range(self.columns):
                estado_actual = self.grid[y][x]
                vecinos_vivos = self.contar_vecinos_vivos(x, y)
                
                if estado_actual == 0 and vecinos_vivos == 3:
                    nueva_grid[y][x] = 1  # nacer
                elif estado_actual == 1 and (vecinos_vivos < 2 or vecinos_vivos > 3):
                    nueva_grid[y][x] = 0  # morir
                else:
                    nueva_grid[y][x] = estado_actual  # el estado que ya tiene
        
        self.grid = nueva_grid

    def contar_vecinos_vivos(self, x, y): #recorre las celdas vecinas para contar vecinos vivos
        vecinos = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),         (0, 1),        #lista de tuplas para los vecinos
            (1, -1), (1, 0), (1, 1)
        ]
        vivos = 0
        for dx, dy in vecinos:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.columns and 0 <= ny < self.rows:
                vivos += self.grid[ny][nx]
        return vivos

    def dibujar_grid(self, screen):
        for y in range(self.rows):
            for x in range(self.columns):
                color = (255, 255, 255) if self.grid[y][x] == 1 else (0, 0, 0)
                pygame.draw.rect(screen, color, (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))

# función para mostrar texto en pantalla
def draw_text(screen, text, font, color, position):
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, position)

# función principal del juego
def main():
    pygame.init()
    screen_width = 800
    screen_height = 600
    cell_size = 10
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Juego de la Vida")

    juego = JuegoDeLaVida(screen_width, screen_height, cell_size)

    black = (0, 0, 0)
    white = (255, 255, 255)
    red = (255, 0, 0)

    font = pygame.font.SysFont(None, 30)  # Fuente para el texto del contador
    iteration_count = 0  # Contador de iteraciones

    running = True
    paused = False  # Estado de pausa
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused  # alternar entre pausa y reanudación

        if not paused:
            juego.actualizar_grid()
            iteration_count += 1  # incrementa el contador de iteraciones

        screen.fill(black)
        juego.dibujar_grid(screen)

        # mostrar el contador de iteraciones en la pantalla e indicar como pausar
        draw_text(screen, f"PARA PAUSAR PRESIONE ESPACIO | Iteraciones: {iteration_count}", font, white, (10, 10))

        # mostrar el estado de pausa
        if paused:
            draw_text(screen, "PAUSADO", font, red, (screen_width // 2 - 50, screen_height // 2))

        pygame.display.flip()
        clock.tick(100)  # controla la velocidad del bucle

    pygame.quit()

if __name__ == "__main__":
    main()