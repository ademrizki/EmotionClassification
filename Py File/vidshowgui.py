import tkinter
import cv2
import time

class App:
    def __init__():
        root = tkinter.Tk()
        root.title('Test Frame')
        source = 'vid.mp4'

        vid = MyVidCapture(source)

        canvas = tkinter.Canvas(root, width = vid.width, height = vid.height)
        canvas.pack()

        btn = tkinter.Button(root, text = 'TEKS')
        btn.pack(anchor = tkinter.CENTER, expand = True)

        root.mainloop()

class MyVidCapture:
    def __init__():
        vid = cv2.VideoCapture('vid.mp4')
        if not vid.isOpened():
            raise ValueError("Unable to open video source", e)
        
        width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

