import numpy as np
import cv2 as cv

img = cv.imread('image/opencvlogo.png')
rows, cols, ch = img.shape
cv.imshow('original image', img)

# resize image
img_resize = cv.resize(img, None, fx=3, fy=2, interpolation=cv.INTER_CUBIC)
cv.imshow('resized image', img_resize)

# image translation
M = np.float32([[1, 0, 100], [0, 1, 50]])
img_translate = cv.warpAffine(img, M, (cols, rows))
cv.imshow('translate image', img_translate)

# image rotation
M = cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 90, 1.5)
img_rotate = cv.warpAffine(img, M, (cols, rows))
cv.imshow('rotate image', img_rotate)

# affine transform
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
M = cv.getAffineTransform(pts1, pts2)
img_affine = cv.warpAffine(img, M, (cols, rows))
cv.imshow('affine image', img_affine)

# perspective transform
pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
M = cv.getPerspectiveTransform(pts1, pts2)
img_perspective = cv.warpPerspective(img, M, (300, 300))
cv.imshow('perspective image', img_perspective)

cv.waitKey()
cv.destroyAllWindows()
