from packages.image import *
from packages.digit import *
from sklearn import datasets

digits = datasets.load_digits()
image = format_image(r"D:\Desktop\Coding\UTEC\Introduction to CS\S14\Laboratory\IntroductionToCS-Project-2\asssets\ocho.png")
# closest_digits = get_closest_digits(image, datasets.load_digits())
# similar_target = get_similar_target(closest_digits, 3)
# print(similar_target)

closest_digits_average: ndarray = get_closest_digits(image, get_average_images(digits))
closest_digit = closest_digits_average[0]
print(f"Soy la inteligencia artificial versión 2 y he detectado que el dígito ingresado corresponde al número {closest_digit}.\n Donde {closest_digit} es un número entre 0 y 9")