import numpy as np
from numpy import asarray
from PIL import Image
from scipy import ndimage

  
def main():
    # Open image
    image_in = Image.open('images/imagem4.jpg')
    # convert image to numpy array    
    # convert to int to allow negative values
    image_np = np.array(image_in).astype(int) 
    # Convolution filter usingo mode reflect
    kernel = np.array([[ 1, 0,-1],
                       [ 0, 0, 0],
                       [-1, 0, 1]])  
    im1 = ndimage.convolve(image_np, kernel, mode='reflect')
    # replace negative values with 0
    im1 = np.where(im1<0, 0, im1)
    # convert array to image usint uint8 type
    image_out = Image.fromarray(im1.astype('uint8'))   
    image_out.save('images/A.jpg')

if __name__ == "__main__":
    main()