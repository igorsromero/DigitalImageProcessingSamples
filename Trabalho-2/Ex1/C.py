import numpy as np
import matplotlib.pyplot as plt
import cv2
filePath = "./Trabalho-2/Imagens/quadrado-branco-no-fundo-preto.png"

img = cv2.imread(filePath, 0)
fourier_amplitude = np.fft.fft2(img)
fourier_fase = np.angle(fourier_amplitude)
plt.subplot(153), plt.imshow(fourier_fase, "gray"), plt.title("Fase")
plt.show()
