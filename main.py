import cv2 as cv
import numpy as np
img1 = cv.imread('image/1233.png')
img2 = cv.imread('image/opencvlogo.png')
rows, cols, channels = img2.shape
roi = img1[0:rows, 0:cols]
img2gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
ret, mask = cv.threshold(img2gray, 100, 255, cv.THRESH_BINARY)
mask_inv = cv.bitwise_not(mask)
img1_bg = cv.bitwise_and(roi, roi, mask = mask_inv)
img2_fg = cv.bitwise_and(img2, img2, mask = mask)

dst = cv.add(img1_bg, img2_fg)

img1[0:rows, 0:cols] = dst
cv.imshow('image1', img1)
cv.imshow('roi', roi)
cv.imshow('image2', img2)
#cv.imshow('image2 gray', img2gray)
cv.imshow('image2 threshold', mask)
#cv.imshow('img2 inverse threshold', mask_inv)
#cv.imshow('img1 background', img1_bg)
cv.imshow('img2 foreground', img2_fg)
cv.imshow('result', dst)
cv.waitKey()
