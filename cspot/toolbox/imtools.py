
# Libs
import tifffile
import numpy as np
from skimage import io as skio

# Functions
def tifread(path):
    return tifffile.imread(path)

def normalize(I):
    m = np.min(I)
    M = np.max(I)
    if M > m:
        return (I-m)/(M-m)
    else:
        return I
    
def im2double(I):
    if I.dtype == 'uint16':
        return I.astype('float64')/65535
    elif I.dtype == 'uint8':
        return I.astype('float64')/255
    elif I.dtype == 'float32':
        return I.astype('float64')
    elif I.dtype == 'float64':
        return I
    else:
        print('returned original image type: ', I.dtype)
        return I
      
def imwrite(I,path):
    skio.imsave(path,I)