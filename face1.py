import cv2
import numpy as np
import face_recognition

imgvirat_bgr = face_recognition.load_image_file('img/known/Virat_Kohli_portrait.jpeg')
imgvirat = cv2.cvtColor(imgvirat_bgr , cv2.COLOR_BGR2RGB)


# # cv2.imshow('bgr' , imgvirat_bgr)
# # cv2.imshow('rgb' , imgvirat_rgb)
# # cv2.waitKey(0)


# face = face_recognition.face_locations(imgvirat)[0]
# copy = imgvirat.copy()

# cv2.rectangle(copy, (face[3], face[0]), (face[1], face[2]),(255,0,255) ,2)
# cv2.imshow('copy' , copy)
# cv2.imshow('virat' , imgvirat)
# cv2.waitKey(0)


train_encode = face_recognition.face_encodings(imgvirat)[0]
# compare image of virat with lebron
test = face_recognition.load_image_file('img/unknown/basketball.jpg')
test = cv2.cvtColor(test , cv2.COLOR_BGR2RGB)
test_encode = face_recognition.face_encodings(test)[0]
print(face_recognition.compare_faces([train_encode], test_encode))