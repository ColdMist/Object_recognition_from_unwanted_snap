'based on the implementation of https://gist.github.com/Munawwar/0efcacfb43827ba3a6bac3356315c419'

import numpy as np
import cv2
from PIL import Image
import skimage.io as imgio
from tf_deblur import *

def getSobel (channel):

    sobelx = cv2.Sobel(channel, cv2.CV_16S, 1, 0, borderType=cv2.BORDER_REPLICATE)
    sobely = cv2.Sobel(channel, cv2.CV_16S, 0, 1, borderType=cv2.BORDER_REPLICATE)
    sobel = np.hypot(sobelx, sobely)

    return sobel;

def findSignificantContours (img, sobel_8u):
    image, contours, heirarchy = cv2.findContours(sobel_8u, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find level 1 contours
    level1 = []
    for i, tupl in enumerate(heirarchy[0]):
        # Each array is in format (Next, Prev, First child, Parent)
        # Filter the ones without parent
        if tupl[3] == -1:
            tupl = np.insert(tupl, 0, [i])
            level1.append(tupl)

    # From among them, find the contours with large surface area.
    significant = []
    tooSmall = sobel_8u.size * 5 / 100 # If contour isn't covering 5% of total area of image then it probably is too small
    for tupl in level1:
        contour = contours[tupl[0]];
        area = cv2.contourArea(contour)
        if area > tooSmall:
            #cv2.drawContours(img, [contour], 0, (0,255,0),2, cv2.LINE_AA, maxLevel=1)
            significant.append([contour, area])

    significant.sort(key=lambda x: x[1])
    return [x[0] for x in significant];

def segment(img):
    #img = cv2.imread(path)
    #print(img)
    '''
    im = Image.open(path)
    pixels = im.getdata()          # get the pixels as a flattened sequence
    black_thresh = 50
    nblack = 0
    for pixel in pixels:
        if pixel < black_thresh:
            nblack += 1
    n = len(pixels)
    '''

 
    blurred = cv2.GaussianBlur(img, (5, 5), 0) # Remove noise
    
    # Edge operator
    sobel = np.max( np.array([ getSobel(blurred[:,:, 0]), getSobel(blurred[:,:, 1]), getSobel(blurred[:,:, 2]) ]), axis=0 )
    
    # Noise reduction trick, from http://sourceforge.net/p/octave/image/ci/default/tree/inst/edge.m#l182
    mean = np.mean(sobel)
    
    # Zero any values less than mean. This reduces a lot of noise.
    sobel[sobel <= mean] = 0;
    sobel[sobel > 255] = 255;
        
    #cv2.imwrite('red_removed_fire/i ('+str(i)+').png', sobel);
    
    sobel_8u = np.asarray(sobel, np.uint8)
        
    # Find contours
    significant = findSignificantContours(img, sobel_8u)
        
    # Mask
    mask = sobel.copy()
    mask[mask > 0] = 0
    cv2.fillPoly(mask, significant, 255)
    # Invert mask
    mask = np.logical_not(mask)
        
    #Finally remove the background
    img[mask] = 0;
        
    #fname = path.split('/')[-1]
    imgio.imsave("deblurred-final_segmented.png", img)
    #cv2.imwrite('seg', img);
    #print(path)
    return img
#for i in range (3,4):
#    segment('new 2/i ('+ str(i)+').jpg')
def deblur_calc():
    img = cv2.imread("deblurred-final.png")
    segmented = segment(img)
