from packages.image import *
from packages.digit import *
from sklearn import datasets

image = format_image(r"D:\Desktop\UTEC\Introduction to CS\S13\Laboratory\cinco.png")
closest_digits = get_closest_digits(image, datasets.load_digits())
print()