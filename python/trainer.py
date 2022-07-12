import cv2
import numpy as np
from PIL import Image
import os

path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier('face/haarcascade_frontalface_default.xml')

def getImagesLabelsWithPath(path):
    Image_Paths = [os.path.join(path, f) for f in os.listdir(path)]
    face_samples = []
    ids = []

    for ImagePath in Image_Paths:
        PIL_img = Image.open(ImagePath).convert('L')
        img_np = np.array(PIL_img, 'uint8')

        id = int(os.path.split(ImagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_np)

        for (x, y, w, h) in faces:
            face_samples.append(img_np[y:y+h, x:x+w])
            ids.append(id)

    return face_samples, ids

print("\n [INFO] Training faces....")

faces, ids = getImagesLabelsWithPath(path)
recognizer.train(faces, np.array(ids))
recognizer.write('trainer/trainer.yml')

print(
    "\n [INFO] {0} Emotions trained. Exiting Program".format(
        len(np.unique(ids)))
    )
