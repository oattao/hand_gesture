import os
import time
import argparse
import cv2 as cv
import numpy as np
import pandas as pd
import mediapipe as mp
from utils.hand import write_landmark_to_csv
from config import ACTIONS

start_time = time.time()

action_labels = dict(zip(ACTIONS, range(len(ACTIONS))))
cordinate = ('x', 'y', 'z')

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

parser = argparse.ArgumentParser(description='Input action name.')
parser.add_argument('--action_name', type=str, choices=ACTIONS)
parser.add_argument('--data_type', type=str, choices=['train', 'test'])
args = parser.parse_args()
action_name = args.action_name
data_type = args.data_type

# prepare folders
data_path = '../data'
data_folder = os.path.join(data_path, data_type)
if not os.path.exists(data_folder):
    os.mkdir(data_folder)

# For webcam input:
cap = cv.VideoCapture(0)

wait_time = 3 # s
record_time = 30 # s
print(f'Start in {wait_time}')
time.sleep(wait_time)

# prepare csv data logfile
num_points = 21

columns = []
for i in range(num_points):
    columns.extend([f'p_{i+1}_{cor}' for cor in cordinate])
datadf = pd.DataFrame(columns=columns)
    
with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv.cvtColor(cv.flip(image, 1), cv.COLOR_BGR2RGB)
    # _, thresh = cv.threshold(image, thresh=125, maxval=255, type=cv.THRESH_BINARY)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    nimage = np.zeros_like(image)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            write_landmark_to_csv(hand_landmarks.landmark, datadf)
            mp_drawing.draw_landmarks(nimage, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    cv.imshow('MediaPipe Hands', nimage)
    cv.imshow('raw', image)
    duration = (time.time() - start_time)
    print()
    if duration >= record_time:
        break
    if cv.waitKey(5) & 0xFF == 27:
        break

cap.release()
datadf['sign'] = action_labels[action_name]
datadf.to_csv(os.path.join(data_folder, f'{action_name}.csv'), index=False)