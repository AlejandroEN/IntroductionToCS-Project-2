from sklearn import datasets
from packages.digit import *
from packages.image import *

digits: Bunch = datasets.load_digits()

# Punto A
average_digits = get_average_images(digits)
print("Matriz con las imágenes promedio de cada dígito creado.\n")

# Punto B
display_average_images(average_digits)
print("Imágenes promedio de cada dígito mostradas.\n")

# Punto C
new_digit_path: str = input("Ingrese la ruta de la imagen del dígito que desea clasificar: ")
new_digit_image: ndarray = format_image(new_digit_path)
print()

# Punto D
closest_digits: ndarray = get_closest_digits(new_digit_image, digits)

# Punto E
print(f"Los 3 dígitos más parecidos al dígito ingresado son: {closest_digits[0]}, {closest_digits[1]} y {closest_digits[2]}\n")

# Punto F
similar_target: tuple[int, int] = get_similar_target(closest_digits, 3)
print(f"Soy la inteligencia artificial y he detectado que el dígito ingresado corresponde al número {similar_target[0]}\nAdemás, utilicé los {similar_target[1]} dígitos más parecidos para llegar a esta conclusión.\n")

# Punto G
closest_digits_average: ndarray = get_closest_digits(new_digit_image, get_average_images(digits))
closest_digit = closest_digits_average[0]
print(f"Soy la inteligencia artificial versión 2 y he detectado que el dígito ingresado corresponde al número {closest_digit}. Donde {closest_digit} es un número entre 0 y 9")