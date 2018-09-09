import cv2
from tf_deblur import *
from segment import *
from utilities import *
import numpy as np

image = cv2.imread('test.png')
reduced= reduce_mb(image)
deblur_calc()

im = cv2.imread('testImage.jpg')
im[np.where((im == [255,182,193]).all(axis = 2))] = [0,33,166]
cv2.imwrite('output.png', im)