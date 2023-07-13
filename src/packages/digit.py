import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray
from sklearn.utils import Bunch

# Punto A
def get_average_images(digits: Bunch) -> Bunch:
    """
    Calcula las imagenes promedio de cada uno de los digitos en el conjunto de datos "digits"
    """
    average_digits: Bunch = Bunch(images=np.zeros((10, 8, 8)), target=np.zeros(10, dtype=int))

    for i in range(10):
        digit_images = digits.images[digits.target == i]
        digit_image_average = np.mean(digit_images, axis=0)
        average_digits.images[i] = digit_image_average
        average_digits.target[i] = i

    return average_digits

# Punto B
def display_average_images(average_digits: Bunch) -> None:
    """
    Muestra las imagenes promedio de cada uno de los digitos en el conjunto de datos "digits"
    """
    for i in range(10):
        plt.gray()
        plt.matshow(average_digits.images[i])
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

    closest_indices = np.argsort(distances)
    closest_digits = digits.target[closest_indices]
    return closest_digits

def get_similar_target(closest_digits: ndarray, n_top: int) -> tuple[int, int]:
    unique_values = np.unique(closest_digits[:n_top], return_counts=True)
    most_recurrent_value_count: int = np.max(unique_values[1])
    most_recurrent_value_index: int = np.where(unique_values[1] == np.max(most_recurrent_value_count))[0][0]

    while most_recurrent_value_count <= len(closest_digits[:n_top])/2:
        return get_similar_target(closest_digits, n_top + 1)
    else:
        return unique_values[0][most_recurrent_value_index], n_top