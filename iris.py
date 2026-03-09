'''
iris.py: Implementa el algoritmo del perceptrón para clasificar el conjunto de datos Iris.
'''

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

def cargar_datos():
    iris = load_iris()
    X = iris.data
    y = iris.target
    print(iris.feature_names)

    return X, y

def main():
    # Cargamos los datos del conjunto de datos Iris
    X, y = cargar_datos()
    print(X.shape)  # Debería mostrar (150, 4)
    
    # Seleccionamos la sepal_lenght y petal_length para la clasificación
    x = X[:, [0, 2]]  # Seleccionamos solo las columnas de sepal_length y petal_length
    print(x)  # Debería mostrar (150, 2)
    print(y)  # 0 = setosa, 1 = versicolor, 2 = virginica

if __name__ == "__main__":
    main()