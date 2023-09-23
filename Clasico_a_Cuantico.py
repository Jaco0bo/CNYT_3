import matplotlib.pyplot as plt
import numpy as np
import vec_and_mat
import math


def simula_canicas(dinamica, inicialstate, clicks):  # Cada vertice tiene una sola flecha(matriz solo con 1 y 0).
    """simula el sistema discreto
    del experimento de las canicas
    (list,list2D, int) -> list2D
    """
    dinamica = np.linalg.matrix_power(dinamica, clicks)
    final_state = vec_and_mat.accion_mv(dinamica, inicialstate)

    return final_state


def probabilistico(dinamica, inicialstate,
                   clicks):  # Para usar esta funcion la matriz dada debe ser doblemente estocastica
    """Simula el experimento las múltiples
    rendijas clásico probabilístico
    (list,list2D, int) -> list2D
    """
    dinamica = np.linalg.matrix_power(dinamica, clicks)
    final_state = vec_and_mat.accion_mv(dinamica, inicialstate)

    return final_state


def cuantico(dinamica, inicialstate, clicks):  # Para este caso la matriz debe contener valores complejos
    """Simula un sistema cuantico
    (list2D, list, int) -> list
    """
    dinamica = np.linalg.matrix_power(dinamica, clicks)
    final_state = vec_and_mat.accion_mv(dinamica, inicialstate)

    return final_state


def grafica(estado):
    """devuelve una grafica que muestra las
    probabilidades de un vector de estados
    (list2D) -> graphic
    """
    fig, ax = plt.subplots()
    caso = []
    for i in range(len(estado)):
        caso.append(i)

    ax.bar(caso, estado, )
    ax.set_ylabel("valores")
    ax.set_title("PROBABILIDAD")
    plt.show()



