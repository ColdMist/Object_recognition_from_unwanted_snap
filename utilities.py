import cv2
import numpy as np
import matplotlib.pyplot as plt

def blur_detect(img):
    img = cv2.imread("test.png")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.Laplacian(gray, cv2.CV_64F).var()
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    text = "Not Blurry"
    if fm < 100:
        text = "Blurry"
    # show the image
    cv2.putText(img, "{}: {:.2f}".format(text, fm), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def color_quantization(img):
    #img = cv2.imread('deblurred-final.png')
    Z = img.reshape((-1, 3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 8
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    cv2.imshow('res2', res2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def calculate_Laplacian():

    img = cv2.imread('deblurred-final.png', 0)

    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

    plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2), plt.imshow(laplacian, cmap='gray')
    plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 3), plt.imshow(sobelx, cmap='gray')
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 4), plt.imshow(sobely, cmap='gray')
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

    plt.show()

def calculate_gradient(img, p_ksize):
    kernel = np.zeros([p_ksize, p_ksize, 3, 3])
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    cv2.imshow('denoised', gradient)
    cv2.waitKey(0)
    return gradient

def denoising_image(img,p_ksize):
    kernel = np.zeros([p_ksize, p_ksize, 3, 3])
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    cv2.imshow('denoised',opening)
    cv2.waitKey(0)
    return opening

def equalize_hist(img):
    for c in range(0, 2):
        img[:, :, c] = cv2.equalizeHist(img[:, :, c])
    cv2.imshow('Histogram equalized', img)
    cv2.waitKey(0)
    return img
