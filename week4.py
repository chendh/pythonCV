import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def show_gray_histogram(img):
    # -- Show Histogram
    cv.imshow('original', img)
    hist = cv.calcHist([img], [0], None, [256], [0, 256])
    plt.plot(hist)
    # ravel 將多維陣列拆開成為一維
    plt.hist(img.ravel(), 256, [0, 256])
    plt.show()


def show_color_histogram(img):
    # -- 畫出彩色影像的直方圖
    # cv.imshow(description, img)
    cv.imshow('original', img)
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        hist = cv.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(hist, color=col)
        plt.xlim([0, 256])
    plt.show()
    pass


def show_histogram_with_subplot(img):
    # -- Show Histogram with subplot
    # 產生一個遮罩
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[100:300, 100:400] = 255
    masked_img = cv.bitwise_and(img, img, mask=mask)
    hist_full = cv.calcHist([img], [0], None, [256], [0, 256])
    hist_mask = cv.calcHist([img], [0], mask, [256], [0, 256])
    plt.subplot(2, 2, 1)
    plt.imshow(img, 'gray')
    plt.subplot(2, 2, 2)
    plt.imshow(mask, 'gray')
    plt.subplot(2, 2, 3)
    plt.imshow(masked_img, 'gray')
    plt.subplot(2, 2, 4)
    plt.plot(hist_full)
    plt.plot(hist_mask)
    plt.xlim([0, 256])
    plt.show()


def opencv_histogram_equalization(img):
    # -- Histogram Equalization with OpenCV
    cv.imshow('Original image', img)
    hist_ori = cv.calcHist([img], [0], None, [256], [0, 256])
    plt.figure(1)
    # plt.plot(hist_ori)
    plt.hist(img.ravel(), 256, [0, 256])  # ravel 將多維陣列拆開成為一維

    img_eq = cv.equalizeHist(img)
    cv.imshow('Equalized image', img_eq)
    hist_eq = cv.calcHist([img_eq], [0], None, [256], [0, 256])
    # plt.plot(hist_eq)
    plt.hist(img_eq.ravel(), 256, [0, 256])
    plt.show()


def image_enhance(img):
    cv.imshow('original', img)
    new_image = np.zeros(img.shape, img.dtype)
    alpha = 1.0
    beta = 50
    gamma = 5
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            for c in range(img.shape[2]):
                # new_image[y, x, c] = np.clip(alpha * img[y, x, c] + beta, 0, 255)
                new_image[y, x, c] = np.clip(pow((img[y, x, c] / 255.0), gamma) * 255.0, 0, 255)
    cv.imshow('Enhanced', new_image)


# Main
# 讀取影像
# print(__name__)
img_ori = cv.imread('image/1233.png')
img_gray = cv.cvtColor(img_ori, cv.COLOR_BGR2GRAY)

image_enhance(img_ori)

# show_gray_histogram(img_gray)

# show_color_histogram(img_ori)

# show_histogram_with_subplot(img_gray)

# opencv_histogram_equalization(img_gray)

cv.waitKey()
