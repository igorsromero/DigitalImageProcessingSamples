
import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, exp

plt.figure(figsize=(6.4 * 5, 4.8 * 5), constrained_layout=False)
imagensPath = "./Trabalho-2/Imagens/"
imagemLena = cv2.imread(imagensPath + "/lena.jpg", 0)


def distancia(point1, point2):
    return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


base = np.ones(imagemLena.shape[:2])
rows, cols = imagemLena.shape[:2]
center = (rows / 2, cols / 2)
for x in range(cols):
    for y in range(rows):
        if distancia((y, x), center) < 50:
            base[y, x] = 0
plt.subplot(152), plt.imshow(np.abs(base),
                             "gray"), plt.title("Filtro Ideal")

base = np.zeros(imagemLena.shape[:2])
rows, cols = imagemLena.shape[:2]
center = (rows / 2, cols / 2)
for x in range(cols):
    for y in range(rows):
        base[y, x] = 1 - 1 / (1 + (distancia((y, x), center) / 50) ** (2 * 20))
plt.subplot(153), plt.imshow(
    np.abs(base), "gray"), plt.title("Filtro Butterworth")

base = np.zeros(imagemLena.shape[:2])
rows, cols = imagemLena.shape[:2]
center = (rows / 2, cols / 2)
for x in range(cols):
    for y in range(rows):
        base[y, x] = 1 - \
            exp(((-distancia((y, x), center) ** 2) / (2 * (50 ** 2))))
plt.subplot(154), plt.imshow(
    np.abs(base), "gray"), plt.title("Filtro Gaussiano")

plt.show()
