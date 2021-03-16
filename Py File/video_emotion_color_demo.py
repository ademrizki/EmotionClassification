import cv2
from keras.models import load_model
from statistics import mode
import numpy as np
from tkinter import Tk
from multiprocessing import Process
from threading import Thread
import tkinter
import time
import sys
import signal
from PIL import Image, ImageTk
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input

# parameters for loading data and images
detection_model_path = 'haarcascade_frontalface_default.xml'
emotion_model_path = 'models/fer2013_mini_XCEPTION.119-0.65.hdf5'
emotion_labels = get_labels('fer2013')

face_id = []

def faceEm():

    # hyper-parameters for bounding boxes shape
    frame_window = 10
    emotion_offsets = (20, 40)

    # loading models
    face_detection = load_detection_model(detection_model_path)
    emotion_classifier = load_model(emotion_model_path, compile=False)

    # getting input model shapes for inference
    emotion_target_size = emotion_classifier.input_shape[1:3]

    # starting lists for calculating modes
    emotion_window = []
    global face_id
    tot_angry = 0
    tot_happy = 0
    tot_neutral = 0
    tot_sad = 0
    tot_fear = 0
    tot_surprise = 0
    tot_disgust = 0

    # starting video streaming
    cv2.namedWindow('GET EMOTION')
    cv2.moveWindow("GET EMOTION", 1500,100)
    video_capture = cv2.VideoCapture(0)
    try:
        while True:
            bgr_image = video_capture.read()[1]
            gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            faces = detect_faces(face_detection, gray_image)
                

            for face_coordinates in faces:

                x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
                gray_face = gray_image[y1:y2, x1:x2]
                try:
                    gray_face = cv2.resize(gray_face, (emotion_target_size))
                except:
                    continue

                gray_face = preprocess_input(gray_face, True)
                gray_face = np.expand_dims(gray_face, 0)
                gray_face = np.expand_dims(gray_face, -1)
                emotion_prediction = emotion_classifier.predict(gray_face)
                emotion_probability = np.max(emotion_prediction)
                emotion_label_arg = np.argmax(emotion_prediction)
                emotion_text = emotion_labels[emotion_label_arg]
                emotion_window.append(emotion_text)

                if len(emotion_window) > frame_window:
                    emotion_window.pop(0)
                try:
                    emotion_mode = mode(emotion_window)
                except:
                    continue


                if emotion_text == 'angry':
                    color = emotion_probability * np.asarray((255, 0, 0))
                    tot_angry += 1
                elif emotion_text == 'sad':
                    color = emotion_probability * np.asarray((0, 0, 255))
                    tot_sad += 1
                elif emotion_text == 'happy':
                    color = emotion_probability * np.asarray((255, 255, 0))
                    tot_happy += 1
                elif emotion_text == 'surprise':
                    color = emotion_probability * np.asarray((0, 255, 255))
                    tot_surprise += 1
                elif emotion_text == 'fear':
                    color = emotion_probability * np.asarray((76, 153, 0))
                    tot_fear += 1
                elif emotion_text == 'disgust':
                    color = emotion_probability * np.asarray((255 , 165, 0))
                    tot_disgust += 1
                else:
                    color = emotion_probability * np.asarray((0, 255, 0))
                    tot_neutral += 1


                color = color.astype(int)
                color = color.tolist()

                total = tot_happy + tot_neutral + tot_angry + tot_sad + tot_surprise + tot_fear + tot_disgust
                per_angry = (tot_angry/total*100)
                per_happy = (tot_happy/total*100)
                per_neutral = (tot_neutral/total*100)
                per_sad = (tot_sad/total*100)
                per_fear = (tot_fear/total*100)
                per_surprise = (tot_surprise/total*100)
                per_disgust = (tot_disgust/total*100)
                per_total = per_angry + per_neutral + per_surprise + per_sad + per_happy + per_fear + per_disgust

                face_id = [ per_angry, per_happy, per_neutral, per_sad, per_fear, per_surprise, per_disgust, per_total]

                draw_bounding_box(face_coordinates, rgb_image, color)
                draw_text(face_coordinates, rgb_image, emotion_mode, color, 0, -10, 1, 1)


            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            res = cv2.resize(bgr_image, (320,240))
            cv2.imshow('GET EMOTION', res)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('\nLagi Senang :' +str(tot_happy)+'    '+str(per_happy))
                print('Lagi Sedih : ' +str(tot_sad)+'    '+str(per_sad))
                print('Lagi Marah :' +str(tot_angry)+'    '+str(per_angry))
                print('Lagi Kaget : ' +str(tot_surprise)+'    '+str(per_surprise))
                print('Lagi Takut : ' +str(tot_fear)+'    '+str(per_fear))
                print('Lagi Jijik : ' +str(tot_disgust)+'    '+str(per_disgust))
                print('Lagi Netral : ' +str(tot_neutral)+'    '+str(per_neutral))
                print('Total Persentasi : ' +str(per_total))
                break
            #return
                
    except KeyboardInterrupt :
        try:
            print('\nLagi Senang :' +str(tot_happy)+'    '+str(per_happy))
            print('Lagi Sedih : ' +str(tot_sad)+'    '+str(per_sad))
            print('Lagi Marah :' +str(tot_angry)+'    '+str(per_angry))
            print('Lagi Kaget : ' +str(tot_surprise)+'    '+str(per_surprise))
            print('Lagi Takut : ' +str(tot_fear)+'    '+str(per_fear))
            print('Lagi Jijik : ' +str(tot_disgust)+'    '+str(per_disgust))
            print('Lagi Netral : ' +str(tot_neutral)+'    '+str(per_neutral))
            print('Total Persentasi : ' +str(per_total))
            print(face_id)
        except NameError:
            print("Tidak Menangkap Emosi")
    except NameError:
        print("Tidak menangkap emosi")

def stimulus():
    try:
        cv2.namedWindow('STIMULUS')
        cv2.moveWindow("STIMULUS", 1500,50)
        vid = cv2.VideoCapture('The secrets to decoding facial expressions.mp4')
        while True:
            ret, frame = vid.read()

            res = cv2.resize(frame, (320,240))
            cv2.imshow('STIMULUS',res)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        vid.release()
        cv2.destroyAllWindows()
    except KeyboardInterrupt:
        print("\nTerinterupsi")

def twoVid():
    global face_id
    Process(target=faceEm).start()
    time.sleep(30)
    Process(target=stimulus).start()

def nStop():
    sys.exit(0)
    

def gui():
    root = Tk()
    root.title("Main Window")

    window_height = 100
    window_width = 500

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    x_cordinate = int((screen_width/2) - (window_width/2))
    y_cordinate = int((screen_height/2) - (window_height/2))

    root.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))
    
    btn1 = tkinter.Button(root, text="Stimulus", command=stimulus)
    btn2 = tkinter.Button(root, text="Stop", command=exit)
    btn3 = tkinter.Button(root, text="Emosi", command=twoVid)

    btn1.grid(row=0, column = 0, pady=50, padx=50)
    btn2.grid(row=0, column = 1, pady=50, padx=50)
    btn3.grid(row=0, column = 2, pady=50, padx=50)

    root.mainloop()
try:
    gui()
except KeyboardInterrupt as e:
    print("\n\nTerinterupsi")    
