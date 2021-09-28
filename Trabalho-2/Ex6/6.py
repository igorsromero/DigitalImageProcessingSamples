import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import difference_of_gaussians, window
from scipy.fftpack import fftn, fftshift

imagensPath = "./Trabalho-2/Imagens/"
imagemLena = cv2.imread(imagensPath + "/lena.jpg", 0)
w_imagemLena = imagemLena * window('hann', imagemLena.shape)
img_filtrada = difference_of_gaussians(imagemLena, 0.5, 50)
w_image_filtrada = img_filtrada * window('hann', imagemLena.shape)
im_fft_mag = fftshift(np.abs(fftn(w_imagemLena)))
fim_fft_mag = fftshift(np.abs(fftn(w_image_filtrada)))

plt.figure(figsize=(6.4 * 5, 4.8 * 5), constrained_layout=False)

plt.subplot(161), plt.imshow(
    imagemLena, "gray"), plt.title("imagemLena Original")
plt.subplot(162), plt.imshow(np.log(im_fft_mag),
                             "gray"), plt.title("Amplitude")
plt.subplot(163), plt.imshow(img_filtrada, "gray"), plt.title("Filtro")
plt.subplot(164), plt.imshow(np.log(fim_fft_mag),
                             "gray"), plt.title("Amplitude e Filtro")

plt.show()
