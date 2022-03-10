import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def image_contrast_enhance(img):
    cv.imshow('gray image', img)
    new_image = np.zeros(img.shape, img.dtype)
    alpha = 1.0
    beta = 50
    gamma = 5
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            for c in range(img.shape[2]):
                # new_image[y, x, c] = np.clip(alpha * img[y, x, c] + beta, 0, 255)
                new_image[y, x, c] = np.clip(pow(img[y, x, c] / 255.0, gamma) * 255.0, 0, 255)
    cv.imshow('Contrast enhanced', new_image)


def show_gray_histogram(img):
    cv.imshow('original gray image', img)
    # hist = cv.calcHist([img], [0], None, [256], [0, 256])
    # plt.plot(hist)
    plt.hist(img.ravel(), 256, [0, 256])
    plt.show()
    pass


def show_color_histogram(img):
    cv.imshow('original color image', img)
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        hist = cv.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(hist, color=col)
        plt.xlim([0, 256])
    plt.show()


def show_histogram_with_subplot(img):
    cv.imshow('original gray image', img)
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[100:300, 200:500] = 255
    masked_image = cv.bitwise_and(img, img, mask=mask)
    cv.imshow('masked image', masked_image)
    hist_full = cv.calcHist([img], [0], None, [256], [0, 256])
    hist_mask = cv.calcHist([img], [0], mask, [256], [0, 256])
    plt.subplot(2, 2, 1)
    plt.imshow(img, 'gray')
    plt.subplot(2, 2, 2)
    plt.imshow(mask, 'gray')
    plt.subplot(2, 2, 3)
    plt.imshow(masked_image, 'gray')
    plt.subplot(2, 2, 4)
    plt.plot(hist_full)
    plt.plot(hist_mask)
    plt.xlim([0, 256])
    plt.show()


def opencv_histogram_equalization(img):
    cv.imshow('original image', img)
    # hist_ori = cv.calcHist([img], [0], None, [256], [0, 256])
    plt.figure(1)
    # plt.plot(hist_ori)
    plt.hist(img.ravel(), 256, [0, 256])
    img_eq = cv.equalizeHist(img)
    cv.imshow('equalized image', img_eq)
    plt.hist(img_eq.ravel(), 256, [0, 256])
    plt.show()


# 讀取影像
img_ori = cv.imread('image/1233.png')
img_gray = cv.cvtColor(img_ori, cv.COLOR_BGR2GRAY)
# cv.imshow('original image', img_ori)
# cv.imshow('gray image', img_gray)
# image_contrast_enhance(img_gray)
# image_contrast_enhance(img_ori)
# show_gray_histogram(img_gray)
# show_color_histogram(img_ori)
# show_histogram_with_subplot(img_gray)
opencv_histogram_equalization(img_gray)
cv.waitKey()
