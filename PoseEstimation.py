#'Lank', = 27
#  'Lelb', = 13
#  'Lhip', = 23
#  'Lkne', = 25
#  'Lsho', = 11
#  'Lwri', = 15
#  'Rank', = 28
#  'Relb', = 14
#  'Rhip', = 24
#  'Rkne', = 26
#  'Rsho', = 12
#  'Rwri', = 16

#go to mediapipe website to see which point corresponds to which number 

import warnings
warnings.filterwarnings('ignore')
try:
    import ujson as json
except ImportError:
    try:
        import simplejson as json
    except ImportError:
        import json
import pickle
import numpy as np
from ipykernel import kernelapp as app
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy import stats
from scipy.integrate import simps
from scipy.spatial import ConvexHull
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

sns.set(font_scale=1.2)

import cv2 as cv
import mediapipe as mp
import time
import numpy as np
import pickle

class poseDetector():

    def __init__(self, mode = False, upBody = False, smooth = True, detectionCon = 0.5, trackCon = 0.5):
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()

    def findPose(self, img, draw = True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        blank = np.zeros(img.shape, dtype='uint8')
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(blank, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return blank

    def findPosition(self, img, draw = True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)   #might be able to get z-values and visibility values if you want
                lmList.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 5, (255, 0, 0), cv.FILLED)
        return lmList
    

def main():
    filename = "Untitled Folder/dancing.mp4"
    cap = cv.VideoCapture(filename)
    pTime = 0
    detector = poseDetector()

    Lank = []
    Lelb = []
    Lhip = []
    Lkne = []
    Lsho = []
    Lwri = []
    Rank = []
    Relb = []
    Rhip = []
    Rkne = []
    Rsho = []
    Rwri = []

    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw=False)

        average = [0, 0]  # change with mean

        try:
            arr = lmList[27]
            arr = arr[1:]
            Lank.append(arr)
        except:
            Lank.append(average)

        try:
            arr = lmList[28]
            arr = arr[1:]
            Rank.append(arr)
        except:
            Rank.append(average)

        try:
            arr = lmList[13]
            arr = arr[1:]
            Lelb.append(arr)
        except:
            Lelb.append(average)

        try:
            arr = lmList[14]
            arr = arr[1:]
            Relb.append(arr)
        except:
            Relb.append(average)
        
        try:
            arr = lmList[23]
            arr = arr[1:]
            Lhip.append(arr)
        except:
            Lhip.append(average)

        try:
            arr = lmList[24]
            arr = arr[1:]
            Rhip.append(arr)
        except:
            Rhip.append(average)

        try:
            arr = lmList[25]
            arr = arr[1:]
            Lkne.append(arr)
        except:
            Lkne.append(average)

        try:
            arr = lmList[26]
            arr = arr[1:]
            Rkne.append(arr)
        except:
            Rkne.append(average)

        try:
            arr = lmList[11]
            arr = arr[1:]
            Lsho.append(arr)
        except:
            Lsho.append(average)

        try:
            arr = lmList[12]
            arr = arr[1:]
            Rsho.append(arr)
        except:
            Rsho.append(average)

        try:
            arr = lmList[15]
            arr = arr[1:]
            Lwri.append(arr)
        except:
            Lwri.append(average)

        try:
            arr = lmList[16]
            arr = arr[1:]
            Rwri.append(arr)
        except:
            Rwri.append(average)

        #print(Rsho) #this gives you all the coordinates at point 14 given in mediapipe website

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        
        cv.putText(img, str(int(fps)), (70,50), cv.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)

        cv.imshow("Image", img) 
        
        if cv.waitKey(1) & 0xFF == ord('q') or success == False:
            break

    position = {"Lank" : Lank, "Lelb" : Lelb, "Lhip" : Lhip, "Lkne" : Lkne, "Lsho" : Lwri, "Lwri" : Lwri, "Rank" : Rank, "Relb" : Relb, "Rhip" : Rhip, "Rkne" : Rkne, "Rsho" : Rwri, "Rwri" : Rwri}
    
    loaded_model = pickle.load(open('Untitled Folder\Lleg_decision_tree_model.sav', 'rb'))
    result = loaded_model.score(position)
    print(result)
    # f = open("data.txt", "w")    
    # # d = {"01" : {'position' : position}}
    # f.write(str(d))
    # f.close()
    # print(position)




    # import csv  

    # header = ["Lank", "Lelb", "Lhip", "Lkne", "Lsho", "Lwri", "Rank", "Relb", "Rhip", "Rkne", "Rsho", "Rwri"]
    # data = [Lank, Lelb, Lhip, Lkne, Lsho, Lwri, Rank, Relb, Rhip, Rkne, Rsho, Rwri]

    

    # with open('text_copy.csv', 'w', encoding='UTF8') as f:
    #     writer = csv.writer(f)

    #     # write the header
    #     writer.writerow(header)

    #     # write the data
    #     writer.writerow(data)

    






if __name__ == "__main__":
    main()