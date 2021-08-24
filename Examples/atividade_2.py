import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def main():
    new_image = 'Images/enhance-me.jpg'

    image = Image.open('Images/enhance-me.gif')

    img_array = np.array(image)

    # Histograma Original
    plt.hist(img_array.flatten(),256,[0,256])
    plt.show()

    #Normalizações e Equalizações
    histogram_array = np.bincount(img_array.flatten(), minlength=256)
    num_pixels = np.sum(histogram_array)
    histogram_array = histogram_array / num_pixels
    chistogram_array = np.cumsum(histogram_array)
    transform_map = np.floor(255 * chistogram_array).astype(np.uint8)

    img_list = list(img_array.flatten())

    eq_img_list = [transform_map[p] for p in img_list]

    eq_img_array = np.reshape(np.asarray(eq_img_list), img_array.shape)

    # Salvando nova imagem
    eq_img = Image.fromarray(eq_img_array)
    eq_img.save(new_image)
    
    # Mostra novo histograma
    equalized_img = Image.open('Images/enhance-me.jpg')
    equalized_array = np.asarray(equalized_img)
    plt.hist(equalized_array.flatten(),256,[0,256])
    plt.show()

    # Processo de aplicar mediana
    image = Image.open(new_image)

    # Imagem para Np Array
    npImage = np.array(image)

    # Aplicando filtro
    m = npImage.shape[0]  # qtd linhas
    n = npImage.shape[1]  # qtd colunas

    for x in range(1, m - 2):
        for y in range(1, n - 2):
            w = npImage[x - 1:x + 2, y - 1:y + 2]
            npImage[x, y] = np.median(w).astype(int)
    
    # Mostrar novo histograma
    plt.hist(npImage.flatten(),256,[0,256])
    plt.show()

    img_median = Image.fromarray(npImage)
    img_median.show()
    img_median.save('Images/enhace-me-median.jpg')


if __name__ == "__main__":
    main()
