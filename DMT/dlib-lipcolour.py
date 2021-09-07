# this code is for changing lip colour , all we need to do is to modify the input and output path respectively

import dlib
import cv2
from imutils import face_utils
import numpy as np
import imutils
import os
from google.colab.patches import cv2_imshow
COLORES =  {
    "color-1":[146, 43, 33],
    'color-2':[205, 97, 85],
    'color-3':[176, 58, 46],
    'color-4':[215, 189, 226],
    'color-5':[229, 152, 102],
    'color-6':[160, 64, 0],
    'color-7':[236, 64, 122],
    'color-8':[244, 143, 177],
    'color-9':[131,18,58],
    'color-10':[205 ,157,174],
    'color-11':[164, 19, 19],
    'color-12':[172, 43, 2],
    'color-13':[135,39,75],}
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
img_path = "/home/ayush-ai/Music/face makeup/Result/faceswap/input"
out_path = "/home/ayush-ai/Music/face makeup/Result/faceswap/lipcolour-output"
for img_name in os.listdir(img_path):
    image = cv2.imread(os.path.join(img_path, img_name))  
    # image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects =face_detector(gray, 1)
    for (i, rect) in enumerate(rects):
        shape = shape_predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        upper_lip = np.array([shape[i-1] for i in [49, 50, 51, 52, 53, 54, 55, 65, 64, 63, 62, 61]])
        lower_lip = np.array([shape[i-1] for i in [49, 60, 59, 58, 57, 56, 55, 65, 66, 67, 68, 61]])
        # cv2.imwrite(os.path.join(out_path, img_name), img)
    for c_name, color_rgb in COLORES.items():
        #print(color)
        img = cv2.fillPoly(image,[upper_lip],color_rgb[::-1])
        img = cv2.fillPoly(image,[lower_lip],color_rgb[::-1])
        out_name = f"{c_name}-{img_name}"
        cv2.imwrite(os.path.join(out_path, out_name), img)