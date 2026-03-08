'''
main.py: Implementa el algoritmo del perceptrón.
'''

from graficar import visualizar_iteracion
import numpy as np

# Función de activación escalón
def escalon(z):
    if z >= 0:
        return 1
    else:
        return 0

def imprimir_actual(w,b,a):
    print("w =", w)
    print("b =", b)
    print("a =", a)

def perceptron(x,w,b,t,a=0.5,epoca=0):
    # Definimos el número de iteraciones para la época actual
    iteracion = 0
    imprimir_actual(w,b,a)
    
    # Calculamos la salida del perceptrón para cada entrada
    for i in range(len(x)):
        # Incrementamos el contador de iteraciones
        iteracion += 1

        # Calculamos la suma ponderada de las entradas
        z = np.dot(x[i],w) + b
        print(f"\tz[{i}] = {x[i]}*{w} + {b} = {z:.2f}")

        # Aplicamos la función de activación escalón
        z_i = escalon(z)
        print(f"\tescalón(z[{i}]) = {z_i}")

        if z_i != t[i][0]:
            # Calculamos el error
            print(f"\t{z_i} != {t[i][0]}\tSalida incorrecta para x[{i}]")      
            e = t[i][0] - z_i 
            # Reajustamos los pesos y el bias
            w = reajustar_pesos(w, a, e, x[i])
            b = reajustar_bias(b, a, e)
            print(f"Reajustamos pesos:")
            imprimir_actual(w,b,a)
            visualizar_iteracion(x,w,b,t,epoca,iteracion,f"Iteración {iteracion}")
    
    return w,b,a

# Numpy tiene la ventaja del broadcasting, 
# Permitiendo realizar operaciones entre arrays y escalare rápidamente.

def reajustar_pesos(w, a, e,x):
    w = w + (a * e * x)
    return w

def reajustar_bias(b, a, e):
    b = b + (a * e)
    return b

def main():
    # Establecemos las entradas de una compuerta
    x = np.array([[0,0],
                  [0,1],
                  [1,0],
                  [1,1]]) 

    # Establecemos los targets de una compuerta AND y OR
    t_AND = np.array([[0],[0],[0],[1]])
    t_OR = np.array([[0],[1],[1],[1]])
    
    # Necesario para el XOR
    t_XOR1 = np.array([[0],[1],[1],[1]])
    t_XOR2 = np.array([[1],[1],[1],[0]])


    # Establecemos los pesos
    w = np.array([0.7, 
                  0.7])
    b = 0.5
    a = 0.5

    convergencia = False

    for i in range(1, 7):
        print(f"\n ===  Época {i} ===")
        w, b, a = perceptron(x,w,b,t_OR,a,epoca=i)
        print(f"Pesos y bias después de la época {i}:")
        imprimir_actual(w,b,a)

if __name__ == "__main__":
    main()