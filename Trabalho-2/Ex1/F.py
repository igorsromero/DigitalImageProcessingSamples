import numpy as np
import matplotlib.pyplot as plt
import cv2

filePath = "./Trabalho-2/Imagens/quadrado-branco-no-fundo-preto.png"
imagensPath = "./Trabalho-2/Imagens/"
imagem = cv2.imread(filePath)
altura, largura = imagem.shape[:2]
# translacao (deslocamento)
deslocamento = np.float32([[1, 0, -50], [0, 1, -90]])
deslocado = cv2.warpAffine(imagem, deslocamento, (largura, altura))
cv2.imshow("Cima e esquerda", deslocado)
cv2.waitKey(0)

cv2.imwrite(imagensPath + '/quadrado_transladada.png', deslocado)

plt.figure(figsize=(6.4*5, 4.8*5), constrained_layout=False)

img_deslocada = cv2.imread(imagensPath + '/quadrado_transladada.png', 0)
plt.subplot(151), plt.imshow(img_deslocada,
                             "gray"), plt.title("Transladada")

img_fft_amplitude = np.fft.fft2(img_deslocada)
plt.subplot(152), plt.imshow(np.log(1+np.abs(img_fft_amplitude)),
                             "gray"), plt.title("Amplitude")

img_fft_fase = np.angle(img_fft_amplitude)
plt.subplot(153), plt.imshow(img_fft_fase, "gray"), plt.title("Fase")

img_centralizado = np.fft.fftshift(img_fft_amplitude)
plt.subplot(154), plt.imshow(np.log(1+np.abs(img_centralizado)),
                             "gray"), plt.title("Centralizado")

plt.show()
