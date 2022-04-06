import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2 as cv
import time
import my_modules


class App:
    def __init__(self, window, window_title, video_source=0):
        self.myfunction = my_modules.CVFunction()
        self.photo = None
        self.window = window
        self.window.title(window_title)
        self.window.geometry('1000x600')
        self.window.config(cursor='arrow')

        self.menubar = tk.Menu(window)

        # 建立一個下拉選單·檔案，然後講臺新增到頂級選單中
        self.file_menu = tk.Menu(self.menubar, tearoff=0)
        self.file_menu.add_command(label="開啟", command=self.myfunction.open_file)
        self.file_menu.add_separator()  # 下拉選單的分隔線
        self.file_menu.add_command(label="退出", command=window.quit)
        self.menubar.add_cascade(label="檔案", menu=self.file_menu)

        # 建立一個功能選單
        self.function_menu = tk.Menu(self.menubar, tearoff=0)
        self.function_menu.add_command(label='Canny Edge Detector', command=self.myfunction.canny_detector)
        self.function_menu.add_command(label='Hough Transform', command=self.myfunction.hough_transform)
        self.menubar.add_cascade(label="功能", menu=self.function_menu)

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
