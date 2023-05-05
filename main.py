# buatlah program yang untuk menampilkan nama dan id dari wajah yang terdeteksi.

import cv2
import numpy as np
import pandas as pd
import os
from PIL import Image

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

path = 'dataset'

# baca database dari file csv
database = pd.read_csv('database.csv')

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

recognizer.read('trainer.yml')

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # tampilkan nama menggunakan database
        nama = database.loc[database['id'] == id]['nama'].values

        # jika confidence < 100 maka tampilkan nama
        if (confidence <= 50):
            nama = nama[0]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            nama = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, nama, (x+5,y-5), font, 1, (255,255,255), 2)

    cv2.imshow('camera',img)

    # jika menekan tombol ESC dan close maka keluar dari program
    k = cv2.waitKey(10) & 0xff
    if k == 27 or k == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

