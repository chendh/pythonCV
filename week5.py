import cv2 as cv
import numpy as np


def my_first_filter(img):
    cv.imshow('original image', img)
    # defining the matrix for kernel to apply filter2d() function on the image to blur the image
    kernelmatrix = np.ones((5, 5), np.float32) / 25
    # applying filter2d() function on the image to blur the image and display it as the output on the screen
    resultimage = cv.filter2D(img, -1, kernelmatrix)
    cv.imshow('Filtered_image', resultimage)


# 讀取影像
img = cv.imread('image/cameraman.png')
my_first_filter(img)
cv.waitKey()
