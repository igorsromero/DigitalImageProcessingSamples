from PIL import Image
img = Image.open('Images/imagem4.jpg').convert('L')
img.save('Images/imagem4.jpg')