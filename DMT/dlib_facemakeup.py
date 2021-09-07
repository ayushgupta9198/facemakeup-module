
#lip color on single color 

import dlib
import cv2
from imutils import face_utils
import numpy as np
import imutils
from google.colab.patches import cv2_imshow
import os
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
img_path = "/home/ayush-ai/Music/face makeup/input1"
out_path = "/home/ayush-ai/Music/face makeup/output"
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
        img = cv2.fillPoly(image,[upper_lip],[20,70,182])
        img = cv2.fillPoly(image,[lower_lip],[20,70,182])
    # cv2.imwrite(f"{out_path}/out-{i}", img)
    cv2.imwrite(os.path.join(out_path, img_name), img)


