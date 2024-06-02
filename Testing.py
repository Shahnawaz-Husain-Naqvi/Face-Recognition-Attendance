import cv2
import numpy as np
import pandas as pd


Date = input('Enter date - dd-mm-yyyy')
face_detect = cv2.CascadeClassifier("C:/cars info/haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read("C:/cars info/trainningData.yml")
id = 0
font = cv2.FONT_HERSHEY_COMPLEX_SMALL


df = pd.read_csv("C:/Users/shahn/Documents/Book12.csv")

while True:
    ret,img = cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray,2,4)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        id,conf = rec.predict(gray[y:y+h,x:x+w])
        id1 = df['Name'][df.Id==id]
        df[Date][df.Id==id] = 'P'
        df.to_csv("C:/Users/shahn/Documents/Book12.csv",index = False)
        cv2.putText(img,str(id1),(x-250,y-9),font,3,255)
    cv2.imshow('face',img)
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break
cam.release()
cv2.destroyAllWindows()

