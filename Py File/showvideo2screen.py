import cv2
import numpy
import time

def stimulus():
    cv2.namedWindow('STIMULUS')
    cv2.moveWindow("STIMULUS", 1500,50)
    vid = cv2.VideoCapture('vid.mp4')
    while True:
        ret, frame = vid.read()

        #res = cv2.resize(frame, (1024,768))
        cv2.imshow('STIMULUS',frame)
        if cv2.waitKey(40) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    vid.release()
    cv2.destroyAllWindows()

stimulus()