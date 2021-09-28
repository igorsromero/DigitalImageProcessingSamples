import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(6.4 * 5, 4.8 * 5), constrained_layout=False)
imagensPath = "./Trabalho-2/Imagens/"
imagemLena = cv2.imread(imagensPath + "/lena.jpg", 0)

plt.subplot(151), plt.imshow(imagemLena, "gray")
plt.show()
