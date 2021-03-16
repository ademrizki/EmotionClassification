import cv2
from keras.models import load_model
from statistics import mode
import numpy as np
from tkinter import Tk
import tkinter
import os
import time
import reportlab
from reportlab.pdfgen import canvas
from multiprocessing import Process
import threading
from threading import Thread
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

per_angry = 0
per_happy = 0
per_neutral = 0
per_sad = 0
per_fear = 0
per_surprise = 0
per_disgust = 0
per_total = 0

nama = input("Masukan Nama: ")
npm = str(input("Masukan NPM: "))

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
    face_id = []
    tot_angry = 0
    tot_happy = 0
    tot_neutral = 0
    tot_sad = 0
    tot_fear = 0
    tot_surprise = 0
    tot_disgust = 0
    global per_angry
    global per_happy
    global per_neutral
    global per_sad
    global per_fear
    global per_surprise
    global per_disgust
    global per_total

    # starting video streaming
    cv2.namedWindow('window_frame')
    cv2.moveWindow("window_frame", 20,20)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('emo_'+str(npm)+'.mp4', fourcc, 20.0, (640,480))
    video_capture = cv2.VideoCapture(0)
    try:
        while True:
            bgr_image = video_capture.read() [1]
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
                    #face_id.append = 1
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
                    color = emotion_probability * np.asarray((255, 105, 180))
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

                draw_bounding_box(face_coordinates, rgb_image, color)
                draw_text(face_coordinates, rgb_image, emotion_mode, color, 0, -10, 1, 1)

            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            out.write(bgr_image)
            res = cv2.resize(bgr_image, (320,240))
            cv2.imshow('window_frame', res)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                try:
                    print('\nLagi Senang :' +str(tot_happy)+'    '+str(per_happy))
                    print('Lagi Sedih : ' +str(tot_sad)+'    '+str(per_sad))
                    print('Lagi Marah :' +str(tot_angry)+'    '+str(per_angry))
                    print('Lagi Kaget : ' +str(tot_surprise)+'    '+str(per_surprise))
                    print('Lagi Takut : ' +str(tot_fear)+'    '+str(per_fear))
                    print('Lagi Jijik : ' +str(tot_disgust)+'    '+str(per_disgust))
                    print('Lagi Netral : ' +str(tot_neutral)+'    '+str(per_neutral))
                    print('Total Persentasi : ' +str(per_total))
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
                    c.drawString(100,300,"Total Persentasi :    "+str(per_total))
                    c.save()
                    #os.system("Lap_"+npm+".pdf")
                    video_capture.release()
                    out.release()
                    cv2.destroyAllWindows()
                    break
                except UnboundLocalError:
                    print("\nTidak Menangkap Emosi")
                    c = canvas.Canvas("Lap_"+npm+".pdf")
                    c.drawString(100,750,"Nama :    "+nama)
                    c.drawString(100,730,"NPM :    "+str(npm))
                    c.drawString(100,600,"Tidak Menangkap Emosi")
                    c.save()
                    #os.system("Lap_"+npm+".pdf")
                    video_capture.release()
                    out.release()
                    cv2.destroyAllWindows()
                    break

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
            c.drawString(100,300,"Total Persentasi :    "+str(per_total))
            c.save()
            video_capture.release()
            out.release()
            cv2.destroyAllWindows()
        except NameError:
            print("Tidak Menangkap Emosi")

def stimulus():
    try:
        cv2.namedWindow('STIMULUS')
        #cv2.moveWindow("STIMULUS", 1500,50)
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
        print("\nTerinterupsi di stimulus")

def dualthread():
    try:
        Thread(target=faceEm).start()
        Thread(target=stimulus).start()
    except KeyboardInterrupt as e:
        print("\n\n\n\n Terinterupsi")
    
    #stimulus()

dualthread()
#faceEm()