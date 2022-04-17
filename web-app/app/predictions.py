import numpy as np
import cv2
import os 
from app import app, APP_ROOT
import tensorflow as tf
import sklearn
import pickle as pkl
import glob
import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import joblib
import mediapipe as mp
import time

temp_path = os.path.join(APP_ROOT, 'temp')

def spiral():
    img_path = os.path.join(temp_path, 'spiral.png')
    img = cv2.imread(img_path)
    plt.imshow(np.asarray(Image.open(img_path)))
    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    model = tf.keras.models.load_model(os.path.join(APP_ROOT, 'spiral_detection.h5'))
    preds = model.predict([img])
    pred = np.argmax(preds[0], axis=0)
    prob = preds[0][pred]
    return pred

def measurePitch(voiceID, f0min, f0max, unit):
    sound = parselmouth.Sound(voiceID) # read the sound
    pitch = call(sound, "To Pitch", 0.0, f0min, f0max)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)#create a praat pitch object
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    localShimmer =  call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    aqpq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11Shimmer =  call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    harmonicity05 = call(sound, "To Harmonicity (cc)", 0.01, 500, 0.1, 1.0)
    hnr05 = call(harmonicity05, "Get mean", 0, 0)
    harmonicity15 = call(sound, "To Harmonicity (cc)", 0.01, 1500, 0.1, 1.0)
    hnr15 = call(harmonicity15, "Get mean", 0, 0)
    harmonicity25 = call(sound, "To Harmonicity (cc)", 0.01, 2500, 0.1, 1.0)
    hnr25 = call(harmonicity25, "Get mean", 0, 0)
    harmonicity35 = call(sound, "To Harmonicity (cc)", 0.01, 3500, 0.1, 1.0)
    hnr35 = call(harmonicity35, "Get mean", 0, 0)
    harmonicity38 = call(sound, "To Harmonicity (cc)", 0.01, 3800, 0.1, 1.0)
    hnr38 = call(harmonicity38, "Get mean", 0, 0)
    return localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, hnr05, hnr15 ,hnr25 ,hnr35 ,hnr38

def speech():
    wave_file = glob.glob(os.path.normpath(temp_path)+"/*.wav")[0]
    sound = parselmouth.Sound(wave_file)
    (localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, hnr05, hnr15 ,hnr25 ,hnr35 ,hnr38) = measurePitch(sound, 75, 1000, "Hertz")
    toPred = pd.DataFrame(np.column_stack(
        [[localJitter], [localabsoluteJitter], [rapJitter], [ppq5Jitter], [localShimmer],
         [localdbShimmer], [apq3Shimmer], [aqpq5Shimmer], [apq11Shimmer], [hnr05], [hnr15],
         [hnr25]]),
                         columns=["Jitter_rel", "Jitter_abs", "Jitter_RAP", "Jitter_PPQ", "Shim_loc", "Shim_dB",
                                  "Shim_APQ3", "Shim_APQ5", "Shi_APQ11", "hnr05", "hnr15",
                                  "hnr25"])
    toPred = toPred.fillna(0)
    print(toPred)
    # loaded_model = pkl.load(open(os.path.join(APP_ROOT, 'trainedModel.sav'), 'rb'))
    loaded_model = joblib.load(os.path.join(APP_ROOT, 'trainedModel.sav'))
    pred = loaded_model.predict(toPred)[0]

    scale, sr = librosa.load(wave_file)
    FRAME_SIZE = 2048
    HOP_SIZE = 512
    S_scale = librosa.stft(scale, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)

    Y_scale = np.abs(S_scale) ** 2

    Y_log_scale = librosa.power_to_db(Y_scale)  #Log-Amplitude Spectrogram
    plot_spectrogram(Y_log_scale, sr, HOP_SIZE)
    plt.savefig(os.path.join(temp_path, "amplitude.png"))

    plot_spectrogram(Y_log_scale, sr, HOP_SIZE, y_axis="log") #Log-Frequency Spectrogram
    plt.savefig(os.path.join(temp_path, "frequency.png"))

    plt.figure(figsize=(15,4))
    data1,sample_rate1 = librosa.load(wave_file, sr=22050, mono=True, offset=0.0, duration=50, res_type='kaiser_best')
    librosa.display.waveshow(data1,sr=sample_rate1, max_points=50000, x_axis='time', offset=0.0)
    plt.savefig(os.path.join(temp_path, "wave.png"))

    return pred

def gait():
    vid_file = glob.glob(os.path.normpath(temp_path)+"/*.mp4")[0]
    cap = cv2.VideoCapture(vid_file)
    # EX: loaded_model = joblib.load(os.path.join(APP_ROOT, 'trainedModel.sav'))
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
        
        # cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
        
        # cv2.imshow("Image", img) 
        
        if cv2.waitKey(1) & 0xFF == ord('q') or success == False:
            break

    
    position = {"Lank" : Lank, "Lelb" : Lelb, "Lhip" : Lhip, "Lkne" : Lkne, "Lsho" : Lwri, "Lwri" : Lwri, "Rank" : Rank, "Relb" : Relb, "Rhip" : Rhip, "Rkne" : Rkne, "Rsho" : Rwri, "Rwri" : Rwri}
    for p in position:
        print(np.array(position[p]).shape)
        print(position[p])
    
    position = [Lank, Lelb, Lhip, Lkne, Lsho, Lwri, Rank, Relb, Rhip, Rkne, Rsho, Rwri]

    loaded_model1 = joblib.load(os.path.join(APP_ROOT, 'Larm_decision_tree_model.sav'))
    loaded_model2 = joblib.load(os.path.join(APP_ROOT, 'Lleg_decision_tree_model.sav'))
    loaded_model3 = joblib.load(os.path.join(APP_ROOT, 'Rarm_decision_tree_model.sav'))
    loaded_model4 = joblib.load(os.path.join(APP_ROOT, 'Rleg_decision_tree_model.sav'))

    results = 0

    results += loaded_model1.predict(position)
    results += loaded_model2.predict(position)
    results += loaded_model3.predict(position)
    results += loaded_model4.predict(position)

    if results / 4 > 2:
        return 1
    else:
        return 0


def plot_spectrogram(Y, sr, hop_length, y_axis="linear"):
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(Y, 
                             sr=sr, 
                             hop_length=hop_length, 
                             x_axis="time", 
                             y_axis=y_axis)
    plt.colorbar(format="%+2.f")

class poseDetector():

    def __init__(self, mode = False, upBody = False, smooth = True, detectionCon = 0.5, trackCon = 0.5):

        # self.mode = mode
        # self.upBody = upBody
        # self.smooth = smooth
        # self.detectionCon = detectionCon
        # self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()

    def findPose(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList