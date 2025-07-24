#quiero hacer un algoritmo de ascencion de colinas que encuentre el maximo de la funcion f(x)= sin(x)/(x+0.1) en el intervalo [-10,-6] 
#con un error de 0.1

import numpy as np
import matplotlib.pyplot as plt
import random as rd
def f(x):
    return np.sin(x)/(x+0.1)

def ascension_colinas(f, x0, h, tol):
    x = x0
    while True:
        x_nuevo = x + h
        if f(x_nuevo) > f(x):
            x = x_nuevo
        else:
            h = -h/2
        if abs(h) < tol:
            break
    return x
#como queremos que encuentre el maximo vamos a crear un bucle para que haga algunas iteraciones
#y se quede con el maximo de todas las iteraciones

x_max = []
for i in range(100):
    x_max.append(ascension_colinas(f, rd.uniform(-10, -6), 0.1, 0.1))
x_max = np.array(x_max)
x_max = x_max[np.argmax(f(x_max))]
print(x_max, f(x_max))

#graficamos la funcion y el maximo encontrado
x = np.linspace(-10, 6, 1000)
plt.plot(x, f(x))
plt.scatter(x_max, f(x_max), color='red')

plt.show()

#el maximo encontrado es x= -0.1 y f(x)= 0.998



