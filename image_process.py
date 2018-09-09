import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.signal import convolve2d as conv2
from skimage.filters import try_all_threshold
from skimage import transform
from skimage import filters
from skimage.restoration import wiener
from scipy.signal import convolve2d

from skimage import color, data, restoration

img = cv2.imread('test.png')
'''
mag = filters.sobel(img.astype("float"))

# show the original image
cv2.imshow("Original", mag)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
'''
kernel = np.ones((5,5), np.uint8)
psf = np.ones((5, 5)) / 25

img = convolve2d(img, psf, 'same')

img_erosion = cv2.erode(img, kernel, iterations=2)
img_dilation = cv2.dilate(img_erosion, kernel, iterations=2)

img += 0.1 * img.std() * np.random.standard_normal(img.shape)
deconvolved_img = restoration.wiener(img, psf, 100)

cv2.imshow('deconvolved image',deconvolved_img)
cv2.imshow('Input', img)
cv2.imshow('Erosion', img_erosion)
cv2.imshow('Dilation', img_dilation)

cv2.waitKey(0)
cv2.destroyAllWindows()
'''
blank_image = np.zeros((img.shape[0],img.shape[1],3))
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
edged = cv2.Canny(thresh,50,200)
cv2.imshow("canny edged image",thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

_,contours, hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
print("Number of contours found = ", len(contours))

def get_contour_areas(contours):
    all_areas = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        all_areas.append(area)

    return  all_areas

print(get_contour_areas(contours))

sorted_contours = sorted(contours,key=cv2.contourArea, reverse= True)

print(get_contour_areas(sorted_contours))

for c in sorted_contours:
    cv2.drawContours(img, [c], -1, (255,0,0), 3)
    cv2.waitKey(0)
    cv2.imshow('Contours', img)

cv2.waitKey(0)
cv2.destroyAllWindows()

for c in contours:
    accuracy = 0.03*cv2.arcLength(c,True)
    approx = cv2.approxPolyDP(c,accuracy,True)
    cv2.drawContours(img, [approx], 0, (0,255,0), 2)
    cv2.imshow("appr", img)

cv2.waitKey(0)
cv2.destroyAllWindows()




