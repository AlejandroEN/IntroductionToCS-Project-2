import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray
from sklearn.utils import Bunch

# Punto A
def get_average_images(digits: Bunch) -> Bunch:
    """
    Calcula las imágenes promedio de cada uno de los dígitos en el conjunto de datos "digits".
    """
    # Objeto Bunch para almacenar las imágenes promedio y las etiquetas
    average_digits: Bunch = Bunch(images=np.zeros((10, 8, 8)), target=np.zeros(10, dtype=int))

    # Calcula el promedio de las imágenes para cada dígito
    for i in range(10):
        # Selecciona las imágenes correspondientes al dígito actual
        digit_images = digits.images[digits.target == i]

        # Calcula el promedio de las imágenes seleccionadas a lo largo del eje 0
        digit_image_average = np.mean(digit_images, axis=0)

        # Almacena el promedio de la imagen y la etiqueta en el objeto Bunch
        average_digits.images[i] = digit_image_average
        average_digits.target[i] = i

    return average_digits


# Punto B
def save_average_images(average_digits: Bunch) -> None:
    """
    Muestra las imágenes promedio de cada uno de los dígitos en el conjunto de datos "digits".
    """
    # Itera sobre cada dígito del 0 al 9
    for i in range(10):
        # Configura la paleta de colores en escala de grises
        plt.gray()

        # Muestra la imagen promedio del dígito actual
        plt.matshow(average_digits.images[i])

        # Construye la ruta de salida para guardar la imagen
        output_path = rf"..\assets\average_image_per_target\average_digit_{i}.png"

        # Guarda la imagen en la ruta de salida
        plt.savefig(output_path)


# Punto D
def get_closest_digits(new_digit: ndarray, digits: Bunch) -> ndarray:
    """
    Calcula los 3 dígitos más cercanos a un nuevo número en el conjunto de datos "digits".
    """
    # Lista para almacenar las distancias entre el nuevo dígito y cada imagen en el conjunto de datos
    distances = []

    # Calcula la distancia entre el nuevo dígito y cada imagen en el conjunto de datos
    for digit in digits.images:
        # Calcula la diferencia entre el nuevo dígito y la imagen actual
        residual = new_digit.flatten() - digit.flatten()

        # Calcula la distancia euclidiana entre el nuevo dígito y la imagen actual
        distance = np.sqrt(np.sum(residual ** 2))

        # Almacena la distancia en la lista de distancias
        distances.append(distance)

    # Obtiene los índices de las distancias ordenadas de menor a mayor
    closest_indices = np.argsort(distances)

    # Obtiene las etiquetas de los dígitos más cercanos usando los índices ordenados
    closest_digits = digits.target[closest_indices]

    return closest_digits

def get_similar_target(closest_digits: ndarray, n_top: int) -> tuple[int, int]:
    """
    Obtiene el valor más recurrente entre los dígitos más cercanos y su recurrencia.
    """
    # Calcula los valores únicos y sus conteos entre los dígitos más cercanos
    unique_values = np.unique(closest_digits[:n_top], return_counts=True)

    # Obtiene el conteo máximo entre los valores únicos
    most_recurrent_value_count: int = np.max(unique_values[1])

    # Obtiene el índice del valor más recurrente
    most_recurrent_value_index: int = np.where(unique_values[1] == np.max(most_recurrent_value_count))[0][0]

    # Si el valor más recurrente aparece al menos la mitad de las veces,
    # se devuelve el valor y el número de elementos considerados.
    while most_recurrent_value_count <= len(closest_digits[:n_top]) / 2:
        return get_similar_target(closest_digits, n_top + 1)
    else:
        return unique_values[0][most_recurrent_value_index], n_top