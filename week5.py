import cv2
import cv2 as cv
import numpy as np


def my_first_filter(img):
    cv.imshow('original image', img)
    # 定義一個平均濾波器
    # kernel = np.ones((5, 5), np.float32) / 25
    # kernel = np.array([
    #     [1, 1, 1],
    #     [1, 1, 1],
    #     [1, 1, 1]
    # ]) / 9
    # 使用filter2d()來進行濾波

    # 定義一個edge detection filter
    # kernel = np.array([
    #     [-1, -1, -1],
    #     [-1, 8, -1],
    #     [-1, -1, -1]
    # ])

    # Emboss filter
    # kernel = np.array([
    #     [-2, -1, 0],
    #     [-1, 1, 1],
    #     [0, 1, 2]
    # ])

    # Sobel filter
    kernel = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    result = cv.filter2D(img, -1, kernel)
    cv.imshow('Filtered_image', result)


def averaging_filter(img):
    cv.imshow('original image', img)

    # averaging blur filter
    img_averaging = cv.blur(img, (11, 11))
    cv.imshow('blur image', img_averaging)


def gaussian_filter(img):
    # Gaussian blur filter
    img_gaussian_blur = cv.GaussianBlur(img, (11, 11), -1)
    cv.imshow('Gaussian blur image', img_gaussian_blur)


def median_filter(img):
    # median filter
    img_median = cv.medianBlur(img, 11)
    cv.imshow('median blur image', img_median)


def sobel_filter(img):
    # Sobel filter
    x = cv.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv.Sobel(img, cv2.CV_16S, 0, 1)

    abs_x = cv.convertScaleAbs(x)  # 轉回uint8
    abs_y = cv.convertScaleAbs(y)
    img_sobel = cv.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)

    cv.imshow('x-direction gradient image', abs_x)
    cv.imshow('y-direction gradient image', abs_y)
    cv.imshow('Sobel image', img_sobel)


def laplacian_filter(img):
    # Laplacian filter
    gray_lap = cv2.Laplacian(img, cv2.CV_16S, ksize=3)
    img_laplacian = cv2.convertScaleAbs(gray_lap)  # 轉回uint8
    cv.imshow('Laplacian image', img_laplacian)


def add_gaussian_noise(img, mean=0, sigma=0.1):
    cv.imshow('original image', img)
    # int -> float (標準化)
    img = img / 255
    # 隨機生成高斯 noise (float + float)
    noise = np.random.normal(mean, sigma, img.shape)
    # noise + 原圖
    img_gaussian = img + noise
    # 所有值必須介於 0~1 之間，超過1 = 1，小於0 = 0
    img_gaussian = np.clip(img_gaussian, 0, 1)
    # 原圖: float -> int (0~1 -> 0~255)
    img_gaussian = np.uint8(img_gaussian * 255)
    # noise: float -> int (0~1 -> 0~255)
    noise = np.uint8(noise * 255)
    cv.imshow('Gaussian noise', noise)
    cv.imshow('noise image', img_gaussian)

    dst = cv2.fastNlMeansDenoising(img_gaussian, None, 10, 10, 7)
    cv.imshow('denoise', dst)
    # median_filter(img_gaussian)


#     return gaussian_out , noise# 讀取影像
img_ori = cv.imread('image/cameraman.png')
# my_first_filter(img_ori)
# averaging_filter(img_ori)
# gaussian_filter(img_ori)
# median_filter(img_ori)
# sobel_filter(img_ori)
# laplacian_filter(img_ori)
add_gaussian_noise(img_ori)
cv.waitKey()
