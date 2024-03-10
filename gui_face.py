import cv2 as cv
import numpy as np
from tensorflow.keras.models import model_from_json
import json
from keras.models import load_model

import warnings 
warnings.filterwarnings('ignore')

emotion_model = load_model('model_face.h5')

cap = cv.VideoCapture(0)
emotion_dict = {0:'Angry',1:'Disgusted',2:'Fear',3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}

while True:
    ret,frame = cap.read()
    frame = cv.resize(frame,(1280,720))
    if not ret:
        break
    face_detector = cv.CascadeClassifier('haarcascades_frontalface_default.xml') 
    gray_frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    
    num_faces = face_detector.detectMultiScale(gray_frame,scaleFactor=1.3,minNeighbors = 5)
    
    for (x,y,w,h) in num_faces:
        cv.rectangle(frame,(x,y-50),(x+w,y+h+10),(0,255,0),4)
        roi_gray_frame = gray_frame[y:y+h,x:x+w]
        cropped_img  = np.expand_dims(np.expand_dims(cv.resize(roi_gray_frame,(48,48)),-1),0)
        
        
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv.putText(frame,emotion_dict[maxindex],(x+5,y-20),cv.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,0)
        
    cv.imshow('Emotion detection',frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv.destroyAllWindows()