import cv2
import numpy as np
from numpy import ndarray

# Punto C
def format_image(path) -> ndarray:
    """
    Formatea una imagen para que sea reconocida por el algoritmo de clasificación.
    """
    # Verifica si el path está entre comillas
    if path[0] == '"':
        path = path[1:-1]

    # Lee la imagen y la convierte a escala de grises
    img_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Redimensiona la imagen a 8x8 píxeles
    new_array = cv2.resize(img_array, (8, 8))

    # Invierte la escala de grises (blanco a negro, negro a blanco)
    for i in range(8):
        for j in range(8):
            new_array[i][j] = 255 - new_array[i][j]

    # Reduce el rango de valores de 0-255 a 0-16
    for i in range(8):
        for j in range(8):
            new_array[i][j] = new_array[i][j] * 16 / 255

    # Convierte el nuevo array en un array NumPy con tipo de dato float32
    return np.asarray(new_array, dtype=np.float32)