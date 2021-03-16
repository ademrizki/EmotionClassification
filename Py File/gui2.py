from tkinter import *
import cv2
import os
import threading
from threading import Thread
import cv2
import multiprocessing
from multiprocessing import Process
import sys
import time

def vidia1():
    vid1 = os.system('python3 vidtest.py')
    

def vidia2():
    vid2 = os.system('python3 vidtest2.py')
    

def trit():
    t1 = Process(target=vidia1)
    t2 = Process(target=vidia2)

    t1.start()
    t2.start()








    #try:
    #    t1.start()
    #    t2.start()
        
    #except t1.terminate() as e:
    #    os.system('pkill -9 vidtest2.py')
    #    print("\n\n\n\n\n      DIE\n\n\n\n")
    #    pass




    #if t1.isAlive() == True:
    #   vid3 =  os.system('python3 vidtest.py')
    #elif lanjut == False:
    #    os.kill()
        

    

root = Tk()
root.wm_title('ROOTZ')
var = IntVar()

R1 = Radiobutton(root, text = 'Vid 1', variable = var, value = 1)
R1.pack(anchor = W)

R2 = Radiobutton(root, text = 'Vid 2', variable = var, value = 2)
R2.pack(anchor = W)

R3 = Radiobutton(root, text = 'Vid 3', variable = var, value = 3)
R3.pack(anchor = W)

Button(root, text = 'RUN', command = (lambda:trit())).pack(pady = 10)
var.set('Vid 1')

root.mainloop()

try:
    Process(target=faceEm).start()
    Process(target=stimulus).start()
except KeyboardInterrupt:
    c = canvas.Canvas("Lap_"+npm+".pdf")
    c.drawString(100,750,"Nama :    "+nama)
    c.drawString(100,730,"NPM :    "+str(npm))
    c.drawString(100,650,"Senang :    "+str(per_happy))
    c.drawString(100,600,"Sedih :    "+str(per_sad))
    c.drawString(100,550,"Marah :    "+str(per_angry))
    c.drawString(100,500,"Kaget :    "+str(per_surprise))
    c.drawString(100,450,"Takut :    "+str(per_fear))
    c.drawString(100,400,"Jijik :    "+str(per_disgust))
    c.drawString(100,350,"Netral :    "+str(per_neutral))
    c.save()