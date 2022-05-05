import cv2 as cv
from tkinter import filedialog
import numpy as np
import math
import random


# this is a comment
class MyFunctions:
    def open_file(self):
        filetypes = (
            ('jpg files', '*.jpg'),
            ('png files', '*.png'),
            ('All files', '*.*')
        )
        filename = filedialog.askopenfilename(
            title='Open a file',
            initialdir='/',
            filetypes=filetypes
        )
        # messagebox.showinfo(title='Selected file', message=filename)
        img = cv.imread(filename)
        return img
        # cv.imshow(filename, img)
        # cv.waitKey()

    def save_file(self):
        pass

    def canny_detector(self):
        max_lowThreshold = 100
        window_name = 'Edge Map'
        title_trackbar = 'Min Threshold:'
        ratio = 3
        kernel_size = 3

        # the callback function for trackbar
        def canny_value_change(val):
            low_threshold = val
            img_blur = cv.blur(src_gray, (3, 3))
            detected_edges = cv.Canny(img_blur, low_threshold, low_threshold * ratio, kernel_size)
            mask = detected_edges != 0
            dst = src * (mask[:, :, None].astype(src.dtype))
            cv.imshow(window_name, dst)

        src = self.open_file()
        src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        cv.imshow('original image', src)
        cv.imshow('graylevel image', src_gray)
        cv.namedWindow(window_name)
        cv.createTrackbar(title_trackbar, window_name, 0, max_lowThreshold, canny_value_change)
        canny_value_change(0)
        cv.waitKey()

    def hough_transform(self):
        src = cv.cvtColor(self.open_file(), cv.COLOR_BGR2GRAY)
        dst = cv.Canny(src, 50, 200, None, 3)

        cv.imshow('canny image', dst)

        # Copy edges to the images that will display the results in BGR
        cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
        cdstP = np.copy(cdst)

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
                cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)

        cv.imshow("Source", src)
        cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
        cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)

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
        src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        cv.imshow('Source', src)

        # cv.imshow('Graylevel image', src_gray)
        ret, src_thresh = cv.threshold(src_gray, 127, 255, cv.THRESH_BINARY)
        cv.imshow('Threshold image', src_thresh)

        contours, hierarchy = cv.findContours(src_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contour_all = cv.drawContours(image=src, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=3)
        cv.imshow('Contours', contour_all)
        cv.waitKey()

    def find_contour(self):
        def contour_threshold_callback(val):
            threshold = val
            canny_output = cv.Canny(src_gray, threshold, threshold * 2)
            contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            # Draw contours
            drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
            for i in range(len(contours)):
                color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
                cv.drawContours(drawing, contours, i, color, 2, cv.LINE_8, hierarchy, 0)
            cv.imshow('Contours', drawing)

        src = self.open_file()
        src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        src_gray = cv.blur(src_gray, (3, 3))

        # create a window and a trackbar
        source_window = 'Source image'
        cv.namedWindow(source_window)
        cv.imshow(source_window, src)
        max_thresh = 255
        thresh = 100
        cv.createTrackbar('Threshold: ', source_window, thresh, max_thresh, contour_threshold_callback)
        contour_threshold_callback(thresh)
        cv.waitKey()

    def bounding_box(self):
        def bounding_box_callback(val):
            threshold = val
            # canny_output = cv.Canny(src_gray, threshold, threshold * 2)
            contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
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
                # cv.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)
            cv.imshow('Contours', drawing)

        src = self.open_file()
        src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        src_gray = cv.blur(src_gray, (3, 3))

        # create a window and a trackbar
        max_thresh = 255
        thresh = 100
        source_window = 'Source image'
        cv.namedWindow(source_window)
        canny_output = cv.Canny(src_gray, thresh, thresh * 2)
        cv.imshow(source_window, src)
        cv.createTrackbar('Threshold: ', source_window, thresh, max_thresh, bounding_box_callback)
        bounding_box_callback(thresh)
        cv.waitKey()

    def convex_hull(self):
        def convex_hull_callback(val):
            threshold = val
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
                cv.drawContours(drawing, contours, i, color)
                cv.drawContours(drawing, hull_list, i, color)
            cv.imshow('Contours', drawing)

        src = self.open_file()
        src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        src_gray = cv.blur(src_gray, (3, 3))

        # create a window and a trackbar
        source_window = 'Source image'
        cv.namedWindow(source_window)
        cv.imshow(source_window, src)
        max_thresh = 255
        thresh = 100
        cv.createTrackbar('Threshold: ', source_window, thresh, max_thresh, convex_hull_callback)
        convex_hull_callback(thresh)
        cv.waitKey()


class MyVideoCapture:
    def __init__(self, video_source=0):
        self.vid = cv.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError('Unable to open video source', video_source)
        self.width = self.vid.get(cv.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                return ret, cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            else:
                return ret, None
