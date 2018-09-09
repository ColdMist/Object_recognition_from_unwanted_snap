'Based on the implementation of https://blog.simiacryptus.com/2016/01/deblurring-with-tensorflow.html'

import tensorflow as tf
#import mahotas
import cv2
from matplotlib import pyplot as plt
import skimage.io as imgio
import numpy as np
import scipy.misc
from utilities import *

p_imgscale = .33
p_ksize = 4
p_kiter = 3

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

def calculate_gradient(img):
    kernel = np.zeros([p_ksize, p_ksize, 3, 3])
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    cv2.imshow('denoised', gradient)
    cv2.waitKey(0)
    return gradient

def denoising_image(img):
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

def reduce_mb(image_path):
    #p_imgfile = 'monkey-02.jpg'
    img_raw = cv2.imread('test.png')
    #img_raw.shape
    img_raw = equalize_hist(img_raw)
    #denoising_image(img_raw)
    img_base = scipy.misc.imresize(img_raw, p_imgscale ) /255.0
    imgio.imsave('base.png', img_base)

    kernel = np.zeros([p_ksize, p_ksize, 3, 3])
    for c in range(0 ,3):
        for xy in range(0 ,p_ksize):
            kernel[xy ,xy ,c ,c] = 1.0 /p_ksize

    v_img = tf.Variable(tf.zeros(img_base.shape), name="Unblurred_Image")
    op_img_resize = tf.reshape(v_img, [-1, img_base.shape[0],
                                       img_base.shape[1], img_base.shape[2]])
    pl_kernel = tf.placeholder("float", shape=kernel.shape, name="Kernel")
    op_init = tf.initialize_all_variables()

    op_convolve = op_img_resize
    for blurStage in range(0 ,p_kiter):
        op_convolve = tf.nn.conv2d(op_convolve, pl_kernel,
                                   strides=[1, 1, 1, 1], padding='SAME')

    with tf.Session() as session:
        session.run(op_init)
        img_blurred = session.run(op_convolve, feed_dict={
            v_img: img_base, pl_kernel: kernel})

    imgio.imsave('blurred.png', img_blurred[0])

    pl_blurredImg = tf.placeholder("float", shape=img_blurred.shape)
    op_loss = tf.reduce_sum(tf.square(op_convolve - pl_blurredImg))
    op_optimize = tf.train.GradientDescentOptimizer(0.5).minimize(op_loss)

    def f_pixel(x):
        return 0 if x< 0 else 1 if x > 1 else x


    f_img = np.vectorize(f_pixel, otypes=[np.float])

    with tf.Session() as session:
        session.run(op_init)
        for epoch in range(0, 50):
            img_deblurred = f_img(session.run(v_img, feed_dict={
                pl_blurredImg: img_blurred, pl_kernel: kernel}))
            #imgio.imsave("deblurred-%s.png" % epoch, img_deblurred)
            for iteration in range(0, 10):
                error = session.run([op_optimize, op_loss], feed_dict={
                    pl_blurredImg: img_blurred, pl_kernel: kernel})[1]
                print("%s/%s = %s" % (epoch, iteration, error))
        img_deblurred = f_img(session.run(v_img, feed_dict={
            pl_blurredImg: img_blurred, pl_kernel: kernel}))
    imgio.imsave("deblurred-final.png", img_deblurred)
    return img_deblurred

