from math import exp, sin

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Ecuación diferencial: y' = -y - sin(x)
def f1(x, y):
    return -y - np.sin(x)

def f2(x, y):
    return x + y

def f3(x,y):
    return -x**2 + np.sin(y)

def f4(x,y):
    return ((6*x)-(3*x*y))/(x**(2)+1)

def f5(x,y):
    return x*np.exp(y)

def f6(x,y):
    return x-y


def graficar_campo_y_solucion(ed, x0, y0, xf):
    xmin = min(x0, xf)-1
    xmax = max(x0, xf)+1
    x = np.linspace(xmin, xmax, 20)
    y = np.linspace(-3, 3, 20)

    X, Y = np.meshgrid(x, y)

    # Pendiente en cada punto
    M = ed(X, Y)

    # Componentes de las flechas
    U = np.ones_like(M)
    V = M

    # Normalizar las flechas para que todas tengan el mismo tamaño
    N = np.sqrt(U**2 + V**2)
    U = U / N
    V = V / N

    # Dibujar el campo direccional
    plt.figure(figsize=(8, 6))
    plt.quiver(X, Y, U, V, angles="xy")

    # Resolver el problema de valor inicial y(0)=1
    sol = solve_ivp(
        ed,
        (x0, xf),
        [y0],
        t_eval=np.linspace(x0, xf, 500)
    )

    # Dibujar la solución
    plt.plot(sol.t, sol.y[0], color="red", linewidth=2, label=f"y({x0})={y0}")

    # Decoración
    plt.title("Campo direccional y solución")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()

    plt.show()



graficar_campo_y_solucion(f1, 0, 1, 5)
graficar_campo_y_solucion(f2, -2, 2, -0.5)
graficar_campo_y_solucion(f3, 0, 1, 1)
graficar_campo_y_solucion(f4, 0, 1, 2)
graficar_campo_y_solucion(f5, 0, 1, 0.5)
graficar_campo_y_solucion(f6, 1, 1, 3)
