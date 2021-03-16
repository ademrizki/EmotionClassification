from tkinter import *
import cv2
import os

def runny():
    os.system('python3 vidtest.py')
    

root = Tk()
root.wm_title('ROOTZ')

stations = 'Vid 1', 'Vid 2', 'Vid 3', 'Vid 4'

f = Frame(relief = RAISED, borderwidth = 5)
var = StringVar()

for station in stations:
    radyo = Radiobutton(f, text = station, variable = var, value = station)
    radyo.pack(side = TOP)

f.pack(pady = 10)
Button(root, text = 'RUN', command = (lambda:runny())).pack(pady = 10)
var.set('Vid 1')

root.mainloop()