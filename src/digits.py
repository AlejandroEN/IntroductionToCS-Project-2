from sklearn import datasets
from sklearn.utils import Bunch
import numpy as np
from numpy import ndarray
import cv2

# Punto a del proyecto
def get_average_image(digits: Bunch):
    """
    Calcula las imagenes promedio de cada uno de los digitos en el conjunto de datos "digits"
    """
    average_digits: ndarray = np.zeros((10, 8, 8))

    for i in range(10):
        digit_images = digits.images[digits.target == i]
        average_digit = np.mean(digit_images, axis=0)
        average_digits[i] = average_digit

    return average_digits

def display_average_images(average_digits):
    """
    Muestra el conjunto de datos de "digits" y se encarga de calcular las imagenes promedios del 0 al 9
    """
    for i in range(10):
        print(f"Imagen promedio del dígito {i}:")
        print(average_digits[i])
        print()


# Punto b del proyecto
# average_digits = get_average_image(digits)

# Convertir las matrices de 8x8 en un arreglo bidimensional
# average_digits_2d = average_digits.reshape(10, -1)

# Guardar los datos en un archivo CSV
# np.savetxt('average_digits.csv', average_digits_2d, delimiter=',')
# print("Archivo 'average_digits.csv' creado exitosamente.")

# Punto c del proyecto
def reformat_image(path):
    """

    """
    img_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (8, 8))

    # Invertir escala
    for i in range(8):
        for j in range(8):
            new_array[i][j] = 255 - new_array[i][j]

    # Reducir de 0 a 255 hacia 0 a 16
    for i in range(8):

            new_array[i][j] = new_array[i][j] * 16 / 255

    return new_array

#Punto d del proyecto

def get_closest_digits(new_digit, digits):
    distances = []
    """
     calcula los 3 dígitos más cercanos a un nuevo número en el conjunto de datos "digits
    """

    for digit in digits.images:
        residual = new_digit - digit.flatten()
        euclidean_distance = np.sqrt(np.sum(residual**2))
        distances.append(euclidean_distance)

    closest_indices = np.argsort(distances)[:3]
    closest_digits = digits.target[closest_indices]

    return closest_digits

"""
calcula la distancia euclidiana entre un nuevo número y cada uno de los dígitos en el conjunto de datos "digits"
"""

new_digit = np.array([[0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0, 0, 0]])

"""
Capturar el nuevo número en formato 8x8
"""

# closest_digits = get_closest_digits(new_digit.flatten(), digits)
"""
Obtener los 3 dígitos más cercanos al nuevo número
"""

#Punto d del proyecto

# print("Los 3 dígitos más cercanos al nuevo número son:", closest_digits)
"""
Imprime los digitos mas cercanos
"""