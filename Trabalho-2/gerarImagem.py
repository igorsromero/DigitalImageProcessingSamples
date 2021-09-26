from PIL import Image

widthImage, heightImage = 512, 512
filePath = "./Trabalho-2/Imagens/quadrado-branco-no-fundo-preto.png"
fileName = (filePath, "PNG")

ObjPillow = Image.new("RGB", (widthImage, heightImage))
setPillow = ObjPillow.load()

for row in range(heightImage):
    for col in range(widthImage):
        color = (000)
        rev_col, rev_row = widthImage - col - 1, heightImage - row - 1
        setPillow[col, row] = color
        setPillow[rev_col, row] = color
        setPillow[col, rev_row] = color
        setPillow[rev_col, rev_row] = color

for row in range(206, 306):
    for col in range(206, 306):
        color = (255, 255, 255)
        setPillow[col, row] = color

ObjPillow.save(*fileName)
rgbToGrey = Image.open(filePath).convert('L')
rgbToGrey.save(filePath)
rgbToGrey.show()
