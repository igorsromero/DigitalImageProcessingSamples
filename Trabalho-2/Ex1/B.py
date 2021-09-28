import numpy as np
import matplotlib.pyplot as plt
import cv2

filePath = "./Trabalho-2/Imagens/quadrado-branco-no-fundo-preto.png"

img = cv2.imread(filePath, 0)
fourier_amplitude = np.fft.fft2(img)
plt.subplot(152), plt.imshow(np.log(1+np.abs(fourier_amplitude)),
                             "gray"), plt.title("Amplitude")
plt.show()
