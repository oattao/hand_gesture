import cv2 as cv
import pickle
import numpy as np
import mediapipe as mp
from config import ACTIONS
from simple_model import SimpleModel

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

action_classes = {i: ACTIONS[i] for i in range(len(ACTIONS))}

def parse_landmark(landmark):
    hand = []
    for point in landmark:
        hand.append(point.x)
        hand.append(point.y)
        hand.append(point.z)
    hand = np.array(hand)
    hand = np.expand_dims(hand, 0)
    return hand

# with open('../model/md.pickle', 'rb') as f:
#     model = pickle.load(f)
model = SimpleModel()

# For webcam input:
cap = cv.VideoCapture(0)
# breakpoint()
with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv.cvtColor(cv.flip(image, 1), cv.COLOR_BGR2RGB)
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
        hand = parse_landmark(hand_landmarks.landmark)
        # breakpoint()
        act = model.predict(hand)[0]
        text = action_classes[act]
        # print('-'*50)
        cv.putText(image, text, (50,50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0),
                   2, cv.LINE_AA)
        # print('Action: ', act)

        mp_drawing.draw_landmarks(
            nimage, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    cv.imshow('MediaPipe Hands', nimage)
    cv.imshow('raw', image)
    if cv.waitKey(5) & 0xFF == 27:
      break
cap.release()