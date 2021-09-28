import matplotlib.pyplot as plt
import cv2
import numpy as np

filePath = "./Trabalho-2/Imagens/quadrado-branco-no-fundo-preto.png"
imagensPath = "./Trabalho-2/Imagens/"

img40 = cv2.imread(filePath)
altura, largura = img40.shape[:2]
cv2.waitKey(0)
ponto = (largura / 2, altura / 2)  # ponto no centro da figura
rotacao = cv2.getRotationMatrix2D(ponto, 40, 1.0)
rotacionado = cv2.warpAffine(img40, rotacao, (largura, altura))
cv2.imshow("Rotacionado 40 graus", rotacionado)
cv2.waitKey(0)
cv2.imwrite(imagensPath + '/quadrado_40.png', rotacionado)
plt.figure(figsize=(6.4*5, 4.8*5), constrained_layout=False)
newImage40 = cv2.imread(imagensPath + '/quadrado_40.png', 0)
plt.subplot(151), plt.imshow(newImage40, "gray"), plt.title("Rotada")

newImage40B = np.fft.fft2(newImage40)
plt.subplot(152), plt.imshow(np.log(1+np.abs(newImage40B)),
                             "gray"), plt.title("Amplitude")
newImage40C = np.angle(newImage40B)
plt.subplot(153), plt.imshow(newImage40C, "gray"), plt.title("Fase")

newImage40D = np.fft.fftshift(newImage40B)
plt.subplot(154), plt.imshow(np.log(1+np.abs(newImage40D)),
                             "gray"), plt.title("Centralizado")
plt.show()
