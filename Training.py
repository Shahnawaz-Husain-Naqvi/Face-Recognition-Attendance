import os
import cv2
import numpy as np
from PIL import Image
recognizern = cv2.face.LBPHFaceRecognizer_create()
path = "C:/face attendence img"

def getImageWithID(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faces = []
    IDs = []
    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L')
        faceNP = np.array(faceImg,'uint8')
        Id = int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNP)
        IDs.append(Id)
        cv2.imshow('training',faceNP)
        cv2.waitKey(100)
    return np.array(IDs),faces
Ids,faces = getImageWithID(path)
recognizern.train(faces,np.array(Ids))
recognizern.save("C:/cars info/trainningData.yml")
cv2.destroyAllWindows()