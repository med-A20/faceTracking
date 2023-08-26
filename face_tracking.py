import cv2 as cv
import numpy as np

# Using VIOLA & JOINES

# importing Module (XML FILE)
face_cascade = cv.CascadeClassifier('./haarcascade_frontalface_default.xml')

# web cam
cap = cv.VideoCapture(0)

# same setting
cap.set(3, 720)
cap.set(4, 480)

while True:
    scss, img = cap.read()
    # convert to gray scale
    grayImg = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    # Detecte Face
    faces = face_cascade.detectMultiScale(grayImg, 1.1, 1)

    # Drawing rectangle around face detected
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (w + x , h + y), (255, 0, 0), 2)

    cv.imshow("output",img)
    # cv2.waitKey(x) slow the video by x
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

