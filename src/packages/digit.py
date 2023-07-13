import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray
from sklearn.utils import Bunch

# Punto A
def get_average_images(digits: Bunch) -> ndarray:
    """
    Calcula las imagenes promedio de cada uno de los digitos en el conjunto de datos "digits"
    """
    average_digits: ndarray = np.zeros((10, 8, 8))

    for i in range(10):
        digit_images = digits.images[digits.target == i]
        digit_image_average = np.mean(digit_images, axis=0)
        average_digits[i] = digit_image_average

    return average_digits

# Punto B
def display_average_images(average_digits: ndarray) -> None:
    """
    Muestra las imagenes promedio de cada uno de los digitos en el conjunto de datos "digits"
    """
    for i in range(10):
        plt.gray()
        plt.matshow(average_digits[i])
        plt.show()

# Punto D
def get_closest_digits(new_digit: ndarray, digits: Bunch) -> ndarray:
    """
    Calcula los 3 dígitos más cercanos a un nuevo número en el conjunto de datos "digits"
    """
    distances = []

    for digit in digits.images:
        residual = new_digit.flatten() - digit.flatten()
        distance = np.sqrt(np.sum(residual ** 2))
        distances.append(distance)

    closest_indices = np.argsort(distances)[:3]
    closest_digits = digits.target[closest_indices]

    return closest_digits