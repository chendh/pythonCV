import cv2 as cv
import numpy as np


def my_first_filter(img):
    cv.imshow('original image', img)
    # 定義一個5*5的平均濾波器
    kernel = np.ones((5, 5), np.float32) / 25
    # 使用filter2d()來進行濾波
    result = cv.filter2D(img, -1, kernel)
    cv.imshow('Filtered_image', result)


# 讀取影像
img = cv.imread('image/cameraman.png')
my_first_filter(img)
cv.waitKey()
