import cv2
import sys
from keras.models import load_model
import numpy as np
import argparse
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.inference import load_image
from utils.preprocessor import preprocess_input

class EmotionClassification:

    image_path = sys.argv[1]
    detection_model_path = '/models/detection_models/haarcascade_frontalface_default.xml'
    emotion_model_path = '/models/fer2013_mini_XCEPTION.119-0.65.hdf5'

    emotion_labels = get_labels('fer2013')

    font = cv2.FONT_HERSHEY_SIMPLEX


    emotion_offsets = (20, 40)
    emotion_offsets = (0, 0)


    face_detection = load_detection_model(detection_model_path)
    emotion_classifier = load_model(emotion_model_path, compile=False)



    emotion_target_size = emotion_classifier.input_shape[1:3]


    image_path = '/images/surprise.jpg'
    rgb_image = load_image(image_path, grayscale=False)
    gray_image = load_image(image_path, grayscale=True)
    gray_image = np.squeeze(gray_image)
    gray_image = gray_image.astype('uint8')

    faces = detect_faces(face_detection, gray_image)
    for face_coordinates in faces:
    

        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]

        try:
        
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

    
    #return emotion_text, emotion_probability
        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
        emotion_text = emotion_labels[emotion_label_arg]
        print ("Emotion:", emotion_text)
        print ("Prob:", emotion_probability)

        if emotion_text == emotion_labels[6]:
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)

        draw_bounding_box(face_coordinates, rgb_image, color)
    
        draw_text(face_coordinates, rgb_image, emotion_text, color, 0, -20, 1, 2)
    
    #print ('Emotion1: %s' % emotion_text)
    #print ('Emotion2: %s' % emotion_text)
    
    #print ('Emotion: %s' % emotion_text)
    #print ('Prob: %s' % emotion_probability) 
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite('/images/predicted_test_image.jpg', bgr_image)
    #img = cv2.imread('home/ebdesk/PROJECT/backend/cv/emotion_detection/EmotionClassification/images/predicted_test_image.jpg')
    #cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
