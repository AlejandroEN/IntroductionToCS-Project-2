import cv2
import numpy as np
from numpy import ndarray

# Punto C
def format_image(path) -> ndarray:
    """
    Formatea una imagen para que sea reconocida por el algoritmo de clasificación
    """
    img_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (8, 8))

    # Invertir escala
    for i in range(8):
        for j in range(8):
            new_array[i][j] = 255 - new_array[i][j]

    # Reducir de 0 a 255 hacia 0 a 16
    for i in range(8):
        for j in range(8):
            new_array[i][j] = new_array[i][j] * 16 / 255

    return np.asarray(new_array, dtype=np.float32)