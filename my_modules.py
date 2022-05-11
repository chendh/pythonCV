import math
from tkinter import filedialog

import cv2 as cv
import numpy as np
import random


class MyFunction:
    def open_file(self):
        filetypes = (
            ('jpg files', '*.jpg'),
            ('png files', '*.png'),
            ('All files', '*.*')
        )

        filename = filedialog.askopenfilename(
            title='Open a file',
            initialdir='/',
            filetypes=filetypes)

        # messagebox.showinfo(title='Selected File', message=filename)
        img = cv.imread(filename)
        # cv.imshow('image', img)
        # cv.waitKey()
        return img

    def canny_detector(self):
        max_threshold = 100
        window_name = 'Edge Map'
        title_trackbar = 'Min Threshold:'
        ratio = 3
        kernel_size = 3

        def canny_value_change(val):
            low_threshold = val
            img_blur = cv.blur(src_gray, (3, 3))
            detected_edges = cv.Canny(img_blur, low_threshold, low_threshold * ratio, kernel_size)
            mask = detected_edges != 0
            dst = src * (mask[:, :, None].astype(src.dtype))
            cv.imshow(window_name, dst)

        src = self.open_file()
        src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        cv.namedWindow(window_name)
        cv.createTrackbar(title_trackbar, window_name, 0, max_threshold, canny_value_change)
        canny_value_change(0)
        cv.waitKey()

    def hough_transform(self):
        src = self.open_file()
        dst = cv.Canny(image=src, threshold1=50, threshold2=200, apertureSize=None, L2gradient=3)
        # Copy edges to the images that will display the results in BGR
        cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
        cdst_p = np.copy(cdst)

        lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)

        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                cv.line(cdst, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)

        linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)

        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                cv.line(cdst_p, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)

        cv.imshow("Source", src)
        cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
        cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdst_p)

        cv.waitKey()

    def corner_Harris(self):
        source_window = 'Source image'
        corners_window = 'Corners detected'
        max_thresh = 255

        def cornerHarris_event_handler(val):
            thresh = val
            # Detector parameters
            blockSize = 2
            apertureSize = 3
            k = 0.04
            # Detecting corners
            dst = cv.cornerHarris(src, blockSize, apertureSize, k)
            # Normalizing
            dst_norm = np.empty(dst.shape, dtype=np.float32)
            cv.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
            dst_norm_scaled = cv.convertScaleAbs(dst_norm)
            # Drawing a circle around corners
            for i in range(dst_norm.shape[0]):
                for j in range(dst_norm.shape[1]):
                    if int(dst_norm[i, j]) > thresh:
                        cv.circle(dst_norm_scaled, (j, i), 5, (0), 2)
            # Showing the result
            cv.namedWindow(corners_window)
            cv.imshow(corners_window, dst_norm_scaled)

        src = cv.cvtColor(self.open_file(), cv.COLOR_BGR2GRAY)

        # Create a window and a trackbar
        cv.namedWindow(source_window)
        thresh = 200  # initial threshold
        cv.createTrackbar('Threshold: ', source_window, thresh, max_thresh, cornerHarris_event_handler)
        cv.imshow(source_window, src)
        cornerHarris_event_handler(thresh)
        cv.waitKey()

    def simple_contour(self):
        src = self.open_file()
        # cv.imshow('Source', src)
        src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(src_gray, 127, 255, 0)
        # cv.imshow('Threshold', thresh)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        contour_all = cv.drawContours(image=src, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=3)
        cv.drawContours(src, contours, 1, (0, 0, 255), 3)
        # contour3 = contours[2]
        # cv.drawContours(src, [contour3], 0, (0, 255, 0), 3)
        cv.imshow('Contours', contour_all)
        cv.waitKey()

    def find_contour(self):
        def contour_thresh_callback(val):
            threshold = val
            # Detect edges using Canny
            canny_output = cv.Canny(src_gray, threshold, threshold * 2)
            # Find contours
            contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            # Draw contours
            drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
            for i in range(len(contours)):
                color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
                cv.drawContours(drawing, contours, i, color, 2, cv.LINE_8, hierarchy, 0)
            # Show in a window
            cv.imshow('Contours', drawing)

        source_window = 'Source image'
        src = self.open_file()
        src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        src_gray = cv.blur(src_gray, (3, 3))

        # Create a window and a trackbar
        cv.namedWindow(source_window)
        cv.imshow(source_window, src)
        max_thresh = 255
        thresh = 100  # initial threshold
        cv.createTrackbar('Canny Threshold: ', source_window, thresh, max_thresh, contour_thresh_callback)
        contour_thresh_callback(thresh)
        cv.waitKey()

    def convex_hull(self):
        def convex_hull_callback(val):
            threshold = val
            # Detect edges using Canny
            canny_output = cv.Canny(src_gray, threshold, threshold * 2)
            # Find contours
            contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            # Find the convex hull object for each contour
            hull_list = []
            for i in range(len(contours)):
                hull = cv.convexHull(contours[i])
                hull_list.append(hull)
            # Draw contours + hull results
            drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
            for i in range(len(contours)):
                color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
                # cv.drawContours(drawing, contours, i, color)
                cv.drawContours(drawing, hull_list, i, color)
            # Show in a window
            cv.imshow('Contours', drawing)

        src = self.open_file()
        # Convert image to gray and blur it
        src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        src_gray = cv.blur(src_gray, (3, 3))
        # Create Window
        source_window = 'Source'
        cv.namedWindow(source_window)
        cv.imshow(source_window, src)
        max_thresh = 255
        thresh = 100  # initial threshold
        cv.createTrackbar('Canny thresh:', source_window, thresh, max_thresh, convex_hull_callback)
        convex_hull_callback(thresh)
        cv.waitKey()

    def bounding_boxes(self):
        def bounding_boxes_callback(val):
            threshold = val
            canny_output = cv.Canny(src_gray, threshold, threshold * 2)
            contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            contours_poly = [None] * len(contours)
            boundRect = [None] * len(contours)
            centers = [None] * len(contours)
            radius = [None] * len(contours)

            for i, c in enumerate(contours):
                contours_poly[i] = cv.approxPolyDP(c, 3, True)
                boundRect[i] = cv.boundingRect(contours_poly[i])
                centers[i], radius[i] = cv.minEnclosingCircle(contours_poly[i])

            drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

            for i in range(len(contours)):
                color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
                cv.drawContours(drawing, contours_poly, i, color)
                cv.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])),
                             (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 2)
                cv.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)

            cv.imshow('Contours', drawing)

        src = self.open_file()
        # Convert image to gray and blur it
        src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        src_gray = cv.blur(src_gray, (3, 3))
        source_window = 'Source'
        cv.namedWindow(source_window)
        cv.imshow(source_window, src)
        max_thresh = 255
        thresh = 100  # initial threshold
        cv.createTrackbar('Canny thresh:', source_window, thresh, max_thresh, bounding_boxes_callback)
        bounding_boxes_callback(thresh)
        cv.waitKey()

    def basic_morphology(self):
        src = self.open_file()
        src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        # convert to binary image
        ret, src_thresh = cv.threshold(src_gray, 127, 255, 0)
        # show the thresholded image
        cv.imshow("Thresholded Image", src_thresh)

        # erosion
        erosion_size = 3
        erosion_element = cv.getStructuringElement(cv.MORPH_RECT, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                                   (erosion_size, erosion_size))
        erosion_dst = cv.erode(src_thresh, erosion_element)
        cv.imshow("Erosion", erosion_dst)

        # dilation
        dilation_size = 3
        dilation_element = cv.getStructuringElement(cv.MORPH_RECT, (2 * dilation_size + 1, 2 * dilation_size + 1),
                                                    (dilation_size, dilation_size))
        dilation_dst = cv.dilate(src_thresh, dilation_element)
        cv.imshow("Dilation", dilation_dst)

        # opening
        opening_size = 3
        opening_element = cv.getStructuringElement(cv.MORPH_RECT, (2 * opening_size + 1, 2 * opening_size + 1),
                                                   (opening_size, opening_size))
        opening_dst = cv.morphologyEx(src_thresh, cv.MORPH_OPEN, opening_element)
        cv.imshow("Opening", opening_dst)

        cv.waitKey()

    def morphological_operations(self):
        # def add_gaussian_noise(img, mean=0, sigma=0.1):
        #     img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        #     cv.imshow('Gray', img_gray)
        #     img_gray = img_gray / 255
        #     noise = np.random.normal(mean, sigma, img_gray.shape)
        #     img_noise = img_gray + noise
        #     img_noise = np.clip(img_noise, 0, 1)
        #     img_noise = np.uint8(img_noise * 255)
        #     noise = np.uint8(noise * 255)
        #     cv.imshow('Gaussian', img_noise)
        #     cv.imshow('noise', noise)
        #     # convert the image to binary image
        #     ret, img_thresh = cv.threshold(img_noise, 100, 255, cv.THRESH_BINARY)
        #     # Show thresholded image
        #     cv.imshow('Thresholded', img_thresh)
        #     return img_thresh

        # optional mapping of values with morphological shapes
        def morph_shape(val):
            if val == 0:
                return cv.MORPH_RECT
            elif val == 1:
                return cv.MORPH_CROSS
            elif val == 2:
                return cv.MORPH_ELLIPSE

        def erosion(val):
            erosion_size = cv.getTrackbarPos(kernel_size, erosion_window)
            erosion_shape = morph_shape(cv.getTrackbarPos(element_shape, erosion_window))

            element = cv.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                               (erosion_size, erosion_size))

            erosion_dst = cv.erode(src, element)
            cv.imshow(erosion_window, erosion_dst)

        def dilatation(val):
            dilatation_size = cv.getTrackbarPos(kernel_size, dilation_window)
            dilation_shape = morph_shape(cv.getTrackbarPos(element_shape, dilation_window))
            element = cv.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                               (dilatation_size, dilatation_size))
            dilatation_dst = cv.dilate(src, element)
            cv.imshow(dilation_window, dilatation_dst)

        max_elem = 2
        max_kernel_size = 21
        element_shape = 'Element:\n 0: Rect \n 1: Cross \n 2: Ellipse'
        kernel_size = 'Kernel size:\n 2n +1'
        erosion_window = 'Erosion Demo'
        dilation_window = 'Dilation Demo'

        src = self.open_file()
        cv.imshow('Source', src)
        # img_thresh = add_gaussian_noise(src)
        cv.namedWindow(erosion_window)
        cv.createTrackbar(element_shape, erosion_window, 0, max_elem, erosion)
        cv.createTrackbar(kernel_size, erosion_window, 0, max_kernel_size, erosion)
        cv.namedWindow(dilation_window)
        cv.createTrackbar(element_shape, dilation_window, 0, max_elem, dilatation)
        cv.createTrackbar(kernel_size, dilation_window, 0, max_kernel_size, dilatation)
        erosion(0)
        dilatation(0)


class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return ret, cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            else:
                return ret, None

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

    # Create a window and pass it to the Application object
