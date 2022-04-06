import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2 as cv
import numpy as np
import math
import time


class App:
    def __init__(self, window, window_title, video_source=0):
        self.photo = None
        self.window = window
        self.window.title(window_title)
        self.window.geometry('1000x600')
        self.window.config(cursor='arrow')

        self.menubar = tk.Menu(window)

        # 建立一個下拉選單·檔案，然後講臺新增到頂級選單中
        self.file_menu = tk.Menu(self.menubar, tearoff=0)
        self.file_menu.add_command(label="開啟", command=self.open_file)
        self.file_menu.add_separator()  # 下拉選單的分隔線
        self.file_menu.add_command(label="退出", command=window.quit)
        self.menubar.add_cascade(label="檔案", menu=self.file_menu)

        # 建立一個功能選單
        self.operator_menu = tk.Menu(self.menubar, tearoff=0)
        self.operator_menu.add_command(label='Canny Edge Detector', command=self.canny_detector)
        self.operator_menu.add_command(label='Hough Transofrm', command=self.hough_transform)
        self.menubar.add_cascade(label="功能", menu=self.operator_menu)

        # 顯示選單
        self.window.config(menu=self.menubar)

        self.video_source = video_source

        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)

        # Create a canvas that can fit the above video source size
        self.canvas = tk.Canvas(window, width=self.vid.width, height=self.vid.height)
        self.canvas.pack()

        # Button that lets the user take a snapshot
        self.btn_snapshot = tk.Button(window, text="Snapshot", width=50, command=self.snapshot)
        self.btn_snapshot.pack(anchor=tk.CENTER, expand=True)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()

        self.window.mainloop()

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

    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        if ret:
            cv.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv.cvtColor(frame, cv.COLOR_RGB2BGR))

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        if ret:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)

    def canny_detector(self):
        max_lowThreshold = 100
        window_name = 'Edge Map'
        title_trackbar = 'Min Threshold:'
        ratio = 3
        kernel_size = 3

        def CannyThreshold(val):
            low_threshold = val
            img_blur = cv.blur(src_gray, (3, 3))
            detected_edges = cv.Canny(img_blur, low_threshold, low_threshold * ratio, kernel_size)
            mask = detected_edges != 0
            dst = src * (mask[:, :, None].astype(src.dtype))
            cv.imshow(window_name, dst)

        src = self.open_file()
        src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        cv.namedWindow(window_name)
        cv.createTrackbar(title_trackbar, window_name, 0, max_lowThreshold, CannyThreshold)
        CannyThreshold(0)
        cv.waitKey()

    def hough_transform(self):
        src = self.open_file()
        dst = cv.Canny(src, 50, 200, None, 3)
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

        cv.waitKey()


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


App(tk.Tk(), "OpenCV with Tkinter GUI")
