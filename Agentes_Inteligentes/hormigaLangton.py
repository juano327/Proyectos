import sys
import pygame

class Hormiga:
    def __init__(self, x, y, direction):
        self.x = x
        self.y = y
        self.direction = direction

    def move(self):
        if self.direction == "up":
            self.y -= 1
        elif self.direction == "down":
            self.y += 1
        elif self.direction == "left":
            self.x -= 1
        elif self.direction == "right":
            self.x += 1

    def turn_left(self):
        if self.direction == "up":
            self.direction = "left"
        elif self.direction == "down":
            self.direction = "right"
        elif self.direction == "left":
            self.direction = "down"
        elif self.direction == "right":
            self.direction = "up"

    def turn_right(self):
        if self.direction == "up":
            self.direction = "right"
        elif self.direction == "down":
            self.direction = "left"
        elif self.direction == "left":
            self.direction = "up"
        elif self.direction == "right":
            self.direction = "down"

def draw_text(screen, text, font, color, position):
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, position)

def main():
    pygame.init()
    screen_width = 800
    screen_height = 600
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Hormiga de Langton")

    grid_size = 10
    grid_width = screen_width // grid_size
    grid_height = screen_height // grid_size
    grid = [[0] * grid_height for _ in range(grid_width)]

    # coloca la hormiga en el centro de la pantalla
    ant = Hormiga(grid_width // 2, grid_height // 2, "up")

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
                    paused = not paused  # alternar entre pausa y reanudación con space

        if not paused:
            ant.move()

            

            current_color = grid[ant.x][ant.y]
            if current_color == 0:
                grid[ant.x][ant.y] = 1
                ant.turn_right()
            else:
                grid[ant.x][ant.y] = 0
                ant.turn_left()

            iteration_count += 1  # incrementa el contador de iteraciones

        screen.fill(black)
        for x in range(grid_width):
            for y in range(grid_height):
                color = white if grid[x][y] == 1 else black
                pygame.draw.rect(screen, color, (x * grid_size, y * grid_size, grid_size, grid_size))

        pygame.draw.rect(screen, red, (ant.x * grid_size, ant.y * grid_size, grid_size, grid_size))

        # mostrar el contador de iteraciones en la pantalla y la indicacion de como poner pausa
        draw_text(screen, f"PARA PAUSAR PRESIONE ESPACIO | Iteraciones: {iteration_count}", font, white, (10, 10))

        # mostrar el estado de pausa
        if paused:
            draw_text(screen, "PAUSADO", font, red, (screen_width // 2 - 50, screen_height // 2))

        pygame.display.flip()

        clock.tick(1000)  # controla la velocidad del bucle

    pygame.quit()

if __name__ == "__main__":
    main()
