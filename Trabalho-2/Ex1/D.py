import numpy as np
import matplotlib.pyplot as plt
import cv2

filePath = "./Trabalho-2/Imagens/quadrado-branco-no-fundo-preto.png"
img = cv2.imread(filePath, 0)
fourier_amplitude = np.fft.fft2(img)
fourier_centralizado = np.fft.fftshift(fourier_amplitude)
plt.subplot(154), plt.imshow(np.log(1+np.abs(fourier_centralizado)),
                             "gray"), plt.title("Centralizado")
plt.show()
