import tkinter as tk
import cv2 as cv
from PIL import Image, ImageTk
import my_modules


class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.window.geometry('1000x600')

        self.my_function = my_modules.MyFunctions()

        # Add a menubar
        self.main_menu = tk.Menu(window)

        # Add file submenu
        self.file_menu = tk.Menu(self.main_menu, tearoff=0)
        self.file_menu.add_command(label='開啟檔案', command=self.my_function.open_file)
        self.file_menu.add_command(label='儲存檔案', command=self.my_function.save_file)
        self.file_menu.add_separator()
        self.file_menu.add_command(label='離開程式', command=window.quit)

        # Add operation submenu
        self.operation_menu = tk.Menu(self.main_menu, tearoff=0)
        self.operation_menu.add_command(label='Canny Edge Detector', command=self.my_function.canny_detector)
        self.operation_menu.add_command(label='Hough Transform', command=self.my_function.hough_transform)
        self.operation_menu.add_command(label='Corner Harris', command=self.my_function.corner_Harris)
        self.operation_menu.add_separator()
        self.operation_menu.add_command(label='Simple Contour', command=self.my_function.simple_contour)
        self.operation_menu.add_command(label='Find Contour', command=self.my_function.find_contour)
        self.operation_menu.add_command(label='Convex Hull', command=self.my_function.convex_hull)
        self.operation_menu.add_command(label='Bounding Box', command=self.my_function.bounding_box)

        # Add submenu to mainmenu
        self.main_menu.add_cascade(label='檔案', menu=self.file_menu)
        self.main_menu.add_cascade(label='功能', menu=self.operation_menu)

        # display menu
        self.window.config(menu=self.main_menu, cursor='circle')

        # add a video source
        self.video_source = video_source

        # open video source
        self.vid = my_modules.MyVideoCapture(self.video_source)

        # create a canvas to display the video content
        self.canvas = tk.Canvas(window, width=self.vid.width, height=self.vid.height)
        self.canvas.pack()

        # create a button to capture the frame
        self.btn_snapshot = tk.Button(window, text='snapshot', width='50', command=self.snapshot)
        self.btn_snapshot.pack(anchor=tk.CENTER, expand=True)

        self.delay = 15
        self.update()
        self.window.mainloop()

    def update(self):
        # get a frame from the video source
        ret, frame = self.vid.get_frame()
        if ret:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.window.after(self.delay, self.update)

    def snapshot(self):
        # get a frame from video source
        ret, frame = self.vid.get_frame()
        if ret:
            cv.imwrite('test.jpg', cv.cvtColor(frame, cv.COLOR_RGB2BGR))


App(tk.Tk(), 'OpenCv with Tkinter GUI')
