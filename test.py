import cv2 as cv
import numpy as np
from numpy import double


def my_first_filter(img):
    # # 定義一個平均濾波器
    kernel = np.ones((7, 7), np.float32) / 49
    # kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) /9

    # 定義一個edge detection filter
    kernel1 = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ])

    # Emboss filter
    kernel2 = np.array([
        [-2, -1, 0],
        [-1, 1, 1],
        [0, 1, 2]
    ])

    img_result = cv.filter2D(img, -1, kernel)
    img_result1 = cv.filter2D(img, -1, kernel1)
    img_result2 = cv.filter2D(img, -1, kernel2)
    cv.imshow('original image', img)
    cv.imshow('result image', img_result)
    cv.imshow('edge detection image', img_result1)
    cv.imshow('emboss image', img_result2)


def averaging_filter(img):
    cv.imshow('original image', img)
    img_averaging = cv.blur(img, (5, 5))
    cv.imshow('blur image', img_averaging)


def median_filter(img):
    cv.imshow('original image', img)
    img_median = cv.medianBlur(img, 7)
    cv.imshow('median blur image', img_median)


def gaussian_filter(img):
    cv.imshow('original image', img)
    img_gaussian_blur = cv.GaussianBlur(img, (11, 11), -1)
    cv.imshow('Gaussian blur image', img_gaussian_blur)


def laplacian_filter(img):
    cv.imshow('original image', img)
    gray_lap = cv.Laplacian(img, cv.CV_16S, ksize=5)
    img_laplacian = cv.convertScaleAbs(gray_lap)
    cv.imshow('Laplacian image', img_laplacian)


def sobel_filter(img):
    cv.imshow('original image', img)
    x = cv.Sobel(img, cv.CV_16S, 1, 0)
    y = cv.Sobel(img, cv.CV_16S, 0, 1)
    abs_x = cv.convertScaleAbs(x)
    abs_y = cv.convertScaleAbs(y)
    img_sobel = cv.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)

    cv.imshow('x-direction gradient image', abs_x)
    cv.imshow('y-direction gradient image', abs_y)
    cv.imshow('sobel image', img_sobel)


def add_gaussian_noise(img):
    cv.imshow('original image', img)
    img = img / 255
    mean = 0
    sigma = 0.1
    noise = np.random.normal(mean, sigma, img.shape)
    img_gaussian = img + noise
    img_gaussian = np.clip(img_gaussian, 0, 1)
    img_gaussian = np.uint8(img_gaussian * 255)
    noise = np.uint8(noise * 255)
    cv.imshow('Gaussian noise', noise)
    cv.imshow('noised image', img_gaussian)
    median_filter(img_gaussian)

    img_result = cv.fastNlMeansDenoising(img_gaussian, None, 10, 10, 7)
    cv.imshow('fast denoise', img_result)


def unsharp_mask(img):
    """Return a sharpened version of the image, using an unsharp mask."""
    # For details on unsharp masking, see:
    # https://en.wikipedia.org/wiki/Unsharp_masking
    # https://homepages.inf.ed.ac.uk/rbf/HIPR2/unsharp.htm
    cv.imshow('original image', img)

    kernel_size = (5, 5)
    scale = 3.0
    img_blur = cv.GaussianBlur(img, kernel_size, 1.0)
    cv.imshow('img_blur image', img_blur)
    cv.imshow('sharpen mask', img - img_blur)

    img_sharpen = float(scale + 1.0) * img - float(scale) * img_blur
    img_sharpen = np.maximum(img_sharpen, np.zeros(img_sharpen.shape))
    img_sharpen = np.minimum(img_sharpen, 255 * np.ones(img_sharpen.shape))
    img_sharpen = img_sharpen.round().astype(np.uint8)

    img_sharpen = cv.addWeighted(img, scale + 1.0, img_blur, -1.0 * scale, 0)
    img_sharpen = np.clip(img_sharpen, 0, 255)

    cv.imshow('sharpened image', img_sharpen)


def bilateral_filter(img):
    cv.imshow('original image', img)
    dst = cv.bilateralFilter(img, 9, 5, 5)
    dst1 = cv.bilateralFilter(img, 9, 100, 100)
    dst2 = cv.bilateralFilter(img, 9, 50, 50)

    cv.imshow('bilateralFilter', dst)
    cv.imshow('bilateralFilter1', dst1)
    cv.imshow('bilateralFilter2', dst2)


# reading original image
img_ori = cv.imread('image/cameraman.png')
# img_ori = cv.imread('image/gray.png')
# my_first_filter(img_ori)
# averaging_filter(img_ori)
# median_filter(img_ori)
# gaussian_filter(img_ori)
# laplacian_filter(img_ori)
# sobel_filter(img_ori)
# add_gaussian_noise(img_ori)
# unsharp_mask(img_ori)
bilateral_filter(img_ori)

cv.waitKey()
