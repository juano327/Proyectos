import numpy as np

class QLearningAgent:
    def __init__(self, num_estados, num_acciones, num_episodios,R,Q, gamma=0.9):
        self.num_estados = num_estados
        self.num_acciones = num_acciones
        self.num_episodios = num_episodios
        self.R = R
        self.Q = Q
        self.gamma = gamma

    def aprendizaje(self):
        for _ in range(self.num_episodios):
            estado = np.random.randint(0,5)
            accion = np.random.choice(np.where(self.R[estado] != -1)[0])
            while True:
                 accion = np.random.choice(np.where(self.R[estado] != -1)[0])

                 Qmax = np.argmax(self.Q[accion])
                 self.Q[estado, accion] = self.R[estado, accion] + self.gamma * Qmax
                 if estado == 5 : break
                 estado = accion
        return self.Q

    def normalizar(self):
        Q_norm = self.Q / np.max(self.Q)
        return Q_norm

def main():
    num_estados = 6
    num_acciones = 6
    Q = np.zeros((num_estados, num_acciones))
    R = np.array([
        [-1,  0,  0, -1, -1, -1  ],
        [ 0, -1, -1,  0, -1, -1  ],
        [ 0, -1, -1,  0,  0, -1  ],
        [-1,  0,  0, -1, -1, 100 ],
        [-1, -1,  0, -1, -1, 100 ],
        [-1, -1, -1,  0,  0, 100 ]
        ])
    gamma=0.9

    num_episodios = 1000

    agente = QLearningAgent(num_estados, num_acciones,num_episodios,R,Q,gamma)
    Q = agente.aprendizaje()
    Q_norm = agente.normalizar()

    print("Q:")
    print(Q)

    print("\nQ normalizada:")
    print(Q_norm)

if __name__ == "__main__":
    main()