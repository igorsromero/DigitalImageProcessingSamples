import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy import ndimage

# Seleciona a imagem
image = Image.open('Images/imagem4.jpg')
# image.show()
# Imagem para array
img_array = np.array(image).astype(int)


## ALTERNATIVA A

# Adicionando os pesos
k = np.array([[0, 0, 0],[0, 1, 0],[0, 0, 0]])
# Realiza a convolução e salva a imagem
ndarray = ndimage.convolve(img_array, k, mode='wrap', cval=0.0)
img_array_convolved = np.array(ndarray)
img_convolved = Image.fromarray(img_array_convolved)
img_convolved.convert('L').save('Images/FiltragemEspacial/A.jpg')

## ALTERNATIVA B

# Adicionando os pesos
k = np.array([[1, 0, -1],[0, 0, 0],[-1, 0, 1]])
# Realiza a convolução e salva a imagem
ndarray = ndimage.convolve(img_array, k, mode='wrap', cval=0.0)
img_array_convolved = np.array(ndarray)
img_convolved = Image.fromarray(img_array_convolved)
img_convolved.convert('L').save('Images/FiltragemEspacial/B.jpg')

## ALTERNATIVA C

# Adicionando os pesos
k = np.array([[0, -1, 0],[-1, 4, -1],[0, -1, 0]])
# Realiza a convolução e salva a imagem
ndarray = ndimage.convolve(img_array, k, mode='wrap', cval=0.0)
img_array_convolved = np.array(ndarray)
img_convolved = Image.fromarray(img_array_convolved)
img_convolved.convert('L').save('Images/FiltragemEspacial/C.jpg')

## ALTERNATIVA D

# Adicionando os pesos
k = np.array([[-1, -1, -1],[-1, 8, -1],[-1, -1, -1]])
# Realiza a convolução e salva a imagem
ndarray = ndimage.convolve(img_array, k, mode='wrap', cval=0.0)
img_array_convolved = np.array(ndarray)
img_convolved = Image.fromarray(img_array_convolved)
img_convolved.convert('L').save('Images/FiltragemEspacial/D.jpg')

## ALTERNATIVA E

# Adicionando os pesos
k = np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]])
# Realiza a convolução e salva a imagem
ndarray = ndimage.convolve(img_array, k, mode='wrap', cval=0.0)
img_array_convolved = np.array(ndarray)
img_convolved = Image.fromarray(img_array_convolved)
img_convolved.convert('L').save('Images/FiltragemEspacial/E.jpg')

## ALTERNATIVA F

# Adicionando os pesos
k = np.array([[1, 1, 1],[1, 1, 1],[1, 1, 1]])
#Multiplicação pela constante caso necessário
m = k*(1/9)
# Realiza a convolução e salva a imagem
ndarray = ndimage.convolve(img_array, m, mode='wrap', cval=0.0)
img_array_convolved = np.array(ndarray)
img_convolved = Image.fromarray(img_array_convolved)
img_convolved.convert('L').save('Images/FiltragemEspacial/F.jpg')

## ALTERNATIVA G

# Adicionando os pesos
k = np.array([[1, 2, 1],[2, 4, 2],[1, 2, 1]])
#Multiplicação pela constante caso necessário
m = k*(1/16)
# Realiza a convolução e salva a imagem
ndarray = ndimage.convolve(img_array, m, mode='wrap', cval=0.0)
img_array_convolved = np.array(ndarray)
img_convolved = Image.fromarray(img_array_convolved)
img_convolved.convert('L').save('Images/FiltragemEspacial/G.jpg')

## ALTERNATIVA H

# Adicionando os pesos
k = np.array([[1, 4, 6, 4, 1],[4, 16, 24, 16, 4],[6, 24, 36, 24, 6],[4, 16, 24, 16, 4],[1, 4, 6, 4, 1]])
#Multiplicação pela constante caso necessário
m = k*(1/256)
# Realiza a convolução e salva a imagem
ndarray = ndimage.convolve(img_array, m, mode='wrap', cval=0.0)
img_array_convolved = np.array(ndarray)
img_convolved = Image.fromarray(img_array_convolved)
img_convolved.convert('L').save('Images/FiltragemEspacial/H.jpg')

## ALTERNATIVA I

# Adicionando os pesos
k = np.array([[1, 4, 6, 4, 1],[4, 16, 24, 16, 4],[6, 24, -476, 24, 6],[4, 16, 24, 16, 4],[1, 4, 6, 4, 1]])
#Multiplicação pela constante caso necessário
m = k*(-1/256)
# Realiza a convolução e salva a imagem
ndarray = ndimage.convolve(img_array, m, mode='wrap', cval=0.0)
img_array_convolved = np.array(ndarray)
img_convolved = Image.fromarray(img_array_convolved)
img_convolved.convert('L').save('Images/FiltragemEspacial/I.jpg')