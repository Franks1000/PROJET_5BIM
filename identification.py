#!/usr/bin/env python
import cv2
import pickle
import numpy as np
import os
a = {}
face_cascade= cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
for root, dirs, files in os.walk("trainner/"):
    for file in files:
        filename, ext = os.path.splitext(file)
        a[filename] = cv2.face.LBPHFaceRecognizer_create()


for k,v in a.items(): 
    v.read(os.path.join("trainner",k+".yml"))

id_image=0
color_info=(255, 255, 255)
color_ko=(0, 0, 255)
color_ok=(0, 255, 0)
label = ""

with open("labels.pickle", "rb") as f:
    og_labels=pickle.load(f)
    labels={v:k for k, v in og_labels.items()}

cap=cv2.VideoCapture("video.mp4")
while True:

    ret, frame=cap.read()
    tickmark=cv2.getTickCount()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray, scaleFactor=1.2,minNeighbors=4, minSize=(50, 50))

    for (x, y, w, h) in faces:
        roi_gray=cv2.resize(gray[y:y+h, x:x+w], (50, 50))

        for key,value in a.items() :
            id_, conf = value.predict(roi_gray)
            #Verifie la confiance
            if conf > 95:
                face_color=color_ko
                face_name="Unknown"
            else:
                #Visage reconnu selon le seuil
                face_color=color_ok
                face_name=str(key)
                break
                
              
        label=face_name+" "+'{:5.2f}'.format(conf)

        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_DUPLEX, 1, color_info, 1, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), face_color, 2)

    fps=cv2.getTickFrequency()/(cv2.getTickCount()-tickmark)
    cv2.putText(frame, "FPS: {:05.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color_info, 2)
    cv2.imshow('L42Project', frame)

    key=cv2.waitKey(1)&0xFF
    if key==ord('q'):
        break
    if key==ord('a'):
        for cpt in range(100):
            ret, frame=cap.read()

cv2.destroyAllWindows()
print("Fin")