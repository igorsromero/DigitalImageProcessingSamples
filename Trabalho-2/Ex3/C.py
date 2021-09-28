import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, exp

plt.figure(figsize=(6.4 * 5, 4.8 * 5), constrained_layout=False)
imagensPath = "./Trabalho-2/Imagens/"
imagemLena = cv2.imread(imagensPath + "/lena.jpg", 0)
image_fft = np.fft.fft2(imagemLena)
centeredImage_fft = np.fft.fftshift(image_fft)

def distancia(point1, point2):
    return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


base = np.ones(imagemLena.shape[:2])
rows, cols = imagemLena.shape[:2]
center = (rows / 2, cols / 2)
for x in range(cols):
    for y in range(rows):
        if distancia((y, x), center) < 50:
            base[y, x] = 0
filtro_ideal = centeredImage_fft * base
img_ideal = np.fft.ifftshift(filtro_ideal)
processada_ideal = np.fft.ifft2(img_ideal)
plt.subplot(161), plt.imshow(np.abs(processada_ideal), "gray"), plt.title("Filtro Ideal")

base = np.zeros(imagemLena.shape[:2])
rows, cols = imagemLena.shape[:2]
center = (rows / 2, cols / 2)
for x in range(cols):
    for y in range(rows):
        base[y, x] = 1 - 1 / (1 + (distancia((y, x), center) / 50) ** (2 * 20))
filtro_butterworth = centeredImage_fft * base
img_butterworth = np.fft.ifftshift(filtro_butterworth)
processada_butterworth = np.fft.ifft2(img_butterworth)
plt.subplot(162), plt.imshow(np.abs(processada_butterworth), "gray"), plt.title("Filtro Butterworth")

base = np.zeros(imagemLena.shape[:2])
rows, cols = imagemLena.shape[:2]
center = (rows / 2, cols / 2)
for x in range(cols):
    for y in range(rows):
        base[y, x] = 1 - \
            exp(((-distancia((y, x), center) ** 2) / (2 * (50 ** 2))))
filtro_gaussiano = centeredImage_fft * base
img_gaussiano = np.fft.ifftshift(filtro_gaussiano)
processada_gaussiano = np.fft.ifft2(img_gaussiano)
plt.subplot(163), plt.imshow(np.abs(processada_gaussiano), "gray"), plt.title("Filtro Gaussiano")


plt.show()