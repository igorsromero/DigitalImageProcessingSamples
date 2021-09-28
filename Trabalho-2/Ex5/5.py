import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, exp


def distancia(point1, point2):
    return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def FiltroIdeal(D0, imagemLenaShape):
    base = np.ones(imagemLenaShape[:2])
    rows, cols = imagemLenaShape[:2]
    center = (rows / 2, cols / 2)
    for x in range(cols):
        for y in range(rows):
            if distancia((y, x), center) < D0:
                base[y, x] = 0
    return base


def FiltroButterworth(D0, imagemLenaShape, n):
    base = np.zeros(imagemLenaShape[:2])
    rows, cols = imagemLenaShape[:2]
    center = (rows / 2, cols / 2)
    for x in range(cols):
        for y in range(rows):
            base[y, x] = 1 - 1 / \
                (1 + (distancia((y, x), center) / D0) ** (2 * n))
    return base


def FiltroGaussiano(D0, imagemLenaShape):
    base = np.zeros(imagemLenaShape[:2])
    rows, cols = imagemLenaShape[:2]
    center = (rows / 2, cols / 2)
    for x in range(cols):
        for y in range(rows):
            base[y, x] = 1 - \
                exp(((-distancia((y, x), center) ** 2) / (2 * (D0 ** 2))))
    return base


plt.figure(figsize=(6.4 * 5, 4.8 * 5), constrained_layout=False)

imagensPath = "./Trabalho-2/Imagens/"
imagemLena = cv2.imread(imagensPath + "/lena.jpg", 0)


imagemLena_fft = np.fft.fft2(imagemLena)
imagemLena_fft_filtro = np.fft.fftshift(imagemLena_fft)

filtro_pb_ideal_1 = FiltroIdeal(1.0, imagemLena.shape)
plt.subplot(152), plt.imshow(np.abs(filtro_pb_ideal_1),
                             "gray"), plt.title("Filtro Ideal 1.0")

filtro_pb_ideal_15 = FiltroIdeal(1.5, imagemLena.shape)
plt.subplot(153), plt.imshow(np.abs(filtro_pb_ideal_1),
                             "gray"), plt.title("Filtro Ideal 1.5")

filtro_pb_ideal_50 = FiltroIdeal(50, imagemLena.shape)
plt.subplot(154), plt.imshow(np.abs(filtro_pb_ideal_50),
                             "gray"), plt.title("Filtro Ideal 50")

plt.show()
plt.figure(figsize=(6.4 * 5, 4.8 * 5), constrained_layout=False)

filtro_pb_Butterworth10 = FiltroButterworth(1.0, imagemLena.shape, 20)
plt.subplot(151), plt.imshow(np.abs(filtro_pb_Butterworth10),
                             "gray"), plt.title("Filtro Butterworth 1.0")

filtro_pb_Butterworth15 = FiltroButterworth(1.5, imagemLena.shape, 20)
plt.subplot(152), plt.imshow(np.abs(filtro_pb_Butterworth15),
                             "gray"), plt.title("Filtro Butterworth 1.5")

filtro_pb_Butterworth50 = FiltroButterworth(50, imagemLena.shape, 20)
plt.subplot(153), plt.imshow(np.abs(filtro_pb_Butterworth50),
                             "gray"), plt.title("Filtro Butterworth 50")

plt.show()
plt.figure(figsize=(6.4 * 5, 4.8 * 5), constrained_layout=False)


filtro_pb_Gaussiano10 = FiltroGaussiano(1.0, imagemLena.shape)
plt.subplot(151), plt.imshow(np.abs(filtro_pb_Gaussiano10),
                             "gray"), plt.title("Filtro Gaussiano 1.0")

filtro_pb_Gaussiano15 = FiltroGaussiano(1.5, imagemLena.shape)
plt.subplot(152), plt.imshow(np.abs(filtro_pb_Gaussiano15),
                             "gray"), plt.title("Filtro Gaussiano 1.5")

filtro_pb_Gaussiano50 = FiltroGaussiano(50, imagemLena.shape)
plt.subplot(153), plt.imshow(np.abs(filtro_pb_Gaussiano50),
                             "gray"), plt.title("Filtro Gaussiano 50")

plt.show()

plt.figure(figsize=(6.4 * 5, 4.8 * 5), constrained_layout=False)


filtro_ideal = imagemLena_fft_filtro * filtro_pb_ideal_1
imagemLena_ideal = np.fft.ifftshift(filtro_ideal)
processada_ideal = np.fft.ifft2(imagemLena_ideal)
plt.subplot(161), plt.imshow(np.abs(processada_ideal),
                             "gray"), plt.title("Filtro Ideal 1.0")

filtro_ideal = imagemLena_fft_filtro * filtro_pb_ideal_15
imagemLena_ideal = np.fft.ifftshift(filtro_ideal)
processada_ideal = np.fft.ifft2(imagemLena_ideal)
plt.subplot(162), plt.imshow(np.abs(processada_ideal),
                             "gray"), plt.title("Filtro Ideal 1.5")

filtro_ideal = imagemLena_fft_filtro * filtro_pb_ideal_50
imagemLena_ideal = np.fft.ifftshift(filtro_ideal)
processada_ideal = np.fft.ifft2(imagemLena_ideal)
plt.subplot(163), plt.imshow(np.abs(processada_ideal),
                             "gray"), plt.title("Filtro Ideal 50")


plt.show()
plt.figure(figsize=(6.4 * 5, 4.8 * 5), constrained_layout=False)

filtro_but = imagemLena_fft_filtro * filtro_pb_Butterworth10
imagemLena_butterworth = np.fft.ifftshift(filtro_but)
processada_butterworth = np.fft.ifft2(imagemLena_butterworth)
plt.subplot(161), plt.imshow(np.abs(processada_butterworth),
                             "gray"), plt.title("Filtro Butterworth 1.0")

filtro_but = imagemLena_fft_filtro * filtro_pb_Butterworth15
imagemLena_butterworth = np.fft.ifftshift(filtro_but)
processada_butterworth = np.fft.ifft2(imagemLena_butterworth)
plt.subplot(162), plt.imshow(np.abs(processada_butterworth),
                             "gray"), plt.title("Filtro Butterworth 1.5")

filtro_but = imagemLena_fft_filtro * filtro_pb_Butterworth50
imagemLena_butterworth = np.fft.ifftshift(filtro_but)
processada_butterworth = np.fft.ifft2(imagemLena_butterworth)
plt.subplot(163), plt.imshow(np.abs(processada_butterworth),
                             "gray"), plt.title("Filtro Butterworth 50")

plt.show()
plt.figure(figsize=(6.4 * 5, 4.8 * 5), constrained_layout=False)

filtro_gaussiano = imagemLena_fft_filtro * filtro_pb_Gaussiano10
imagemLena_gaussiano = np.fft.ifftshift(filtro_gaussiano)
processada_gaussiano = np.fft.ifft2(imagemLena_gaussiano)
plt.subplot(161), plt.imshow(np.abs(processada_gaussiano),
                             "gray"), plt.title("Filtro Gaussiano 1.0")

filtro_gaussiano = imagemLena_fft_filtro * filtro_pb_Gaussiano15
imagemLena_gaussiano = np.fft.ifftshift(filtro_gaussiano)
processada_gaussiano = np.fft.ifft2(imagemLena_gaussiano)
plt.subplot(162), plt.imshow(np.abs(processada_gaussiano),
                             "gray"), plt.title("Filtro Gaussiano 1.5")

filtro_gaussiano = imagemLena_fft_filtro * filtro_pb_Gaussiano50
imagemLena_gaussiano = np.fft.ifftshift(filtro_gaussiano)
processada_gaussiano = np.fft.ifft2(imagemLena_gaussiano)
plt.subplot(163), plt.imshow(np.abs(processada_gaussiano),
                             "gray"), plt.title("Filtro Gaussiano 50")

plt.show()
plt.figure(figsize=(6.4 * 5, 4.8 * 5), constrained_layout=False)
