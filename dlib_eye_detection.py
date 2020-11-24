import numpy as np
import cv2
import dlib
import time
from math import *

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('C:/Users/lucas/Documents/IOGS/3A/Projet_Parallax_3D/Test/shape_predictor_68_face_landmarks.dat') #entrainement
cap = cv2.VideoCapture(0)

while(True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    if len(rects)>0:  # Si aucun point n'est détecté
        shape = predictor(gray, rects[0])
        shape = shape_to_np(shape)

    x_mean = floor((shape[39][0] + shape[42][0]) / 2) 
    y_mean = floor((shape[39][1] + shape[42][1]) / 2)
    M = (x_mean,y_mean)

    # -- Visualisation des cercles -- #
    #voir avec bout du nez
    #voir améliorations avec pupilles
    cv2.circle(img, (x_mean, y_mean), 2, (0, 255, 0), -1)
    cv2.imshow('Placement des yeux', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    ## _______________________________ ##

    # Récupérer la résolution de la webcam, la nomrliser 
    #Résoudre le problème de crash si il ne détecte plus de point


    
    
cap.release()
cv2.destroyAllWindows()