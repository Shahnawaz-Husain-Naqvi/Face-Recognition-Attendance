import pandas as pd

df = pd.read_csv("C:/Users/shahn/Documents/Book12.csv")

import cv2

face_cascade = cv2.CascadeClassifier("C:/cars info/haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)

Id = input('enter your id')
Name = input('enter yopur name')
df2 = pd.DataFrame({'Id': [Id] , 'Name': [Name]})
df = pd.concat([df,df2]).drop_duplicates().reset_index(drop = True)
df.to_csv("C:/Users/shahn/Documents/Book12.csv",index = False)
sampleNum = 0

while (True):
    ret,img = cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(144,1,1),3)
        sampleNum +=1 
        new_path = "C:/face attendence img/user." + str(Id) +'.'+ str(sampleNum) + ".jpg" +".jpg"
        cv2.imwrite(new_path,img[y:y+h,x:x+w])
        cv2.imshow('frame',img)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    elif sampleNum > 10:
        break
cam.release()
cv2.destroyAllWindows()