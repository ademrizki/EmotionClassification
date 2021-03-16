import cv2
import numpy as np
import time

cap = cv2.VideoCapture('vid.mp4')

def make480p():
    cap.set(3,640)
    cap.set(4,480)

def rescale_frame(frame, percent=75):
    scale_percent = 50
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width,height)
    return cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

while (True):
    ret, frame = cap.read(0)
    frame = rescale_frame(frame, percent = 30)
    #rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    cv2.imshow('VIDEO TEST', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
