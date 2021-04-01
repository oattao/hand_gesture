import os
import time
import pickle
import webbrowser
from multiprocessing import Process, Pool
import cv2 as cv
import numpy as np

import mediapipe as mp
import pyautogui as pg 

from config import ACTIONS

INTRO_TIME = 3
W, H = pg.size()
EXIT_BUTTON = [W-10, 20]
x_m, y_m = W//2, H//2

def parse_landmark(landmark):
    hand = []
    for point in landmark:
        hand.append(point.x)
        hand.append(point.y)
        hand.append(point.z)
    hand = np.array(hand)
    hand = np.expand_dims(hand, 0)
    return hand

def intro_game():
    screen_intro = np.zeros((H//2, W//2, 3))
    text = 'Game start in {}s'
    font_name = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 3
    font_thickness = 4

    textsize = cv.getTextSize(text, font_name, font_scale, font_thickness)[0]
    xt = (screen_intro.shape[1] - textsize[0]) // 2
    yt = (screen_intro.shape[0] - textsize[1]) // 2
    for i in range(INTRO_TIME):
        screen = screen_intro.copy()
        cv.putText(screen, text.format(INTRO_TIME-i), (xt, yt), font_name, font_scale, (0, 255, 0), font_thickness)
        cv.imshow('Welcome to dino game', screen)
        cv.waitKey(1000)

def launch_game():
    URL = 'chrome:dino'
    chromepath= 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s'
    controler = webbrowser.get(chromepath)
    controler.open(URL, new=0, autoraise=True)

    pg.click(x=211, y=76, button='right')
    pg.write('chrome://dino')
    pg.press('enter')
    resume_game()

def resume_game():
    pg.click(x=x_m, y=y_m, button='right')
    # pg.press('space')

def pause_game():
    pg.click(x=-100, y=y_m, button='right')

def exit_game():
    pause_game()
    time.sleep(1)
    pg.click(x=EXIT_BUTTON[0], y=EXIT_BUTTON[1], button='right')

def play_game():
    action_classes = {i: ACTIONS[i] for i in range(len(ACTIONS))}
    with open('../model/md.pickle', 'rb') as f:
        model = pickle.load(f)

    cap = cv.VideoCapture(0)
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(max_num_hands=1,
                        min_detection_confidence=0.6,
                        min_tracking_confidence=0.6) as hands:
        while cap.isOpened():                                             
            success, image = cap.read()
            if not success:
                continue

            image = cv.cvtColor(cv.flip(image, 1), cv.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to pass by reference.
            image.flags.writeable = False
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
            nimage = np.zeros_like(image)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    hand = parse_landmark(hand_landmarks.landmark)
                    act = model.predict(hand)[0]
                    text = action_classes[act]

                    if text == 'jump':
                        pg.press('space')

                    cv.putText(image, text, (50,50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv.LINE_AA)

                mp_drawing.draw_landmarks(
                    nimage, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv.imshow('MediaPipe Hands', nimage)
            cv.imshow('raw', image)
            if cv.waitKey(5) & 0xFF == 27:
                break

def main():
    intro_game()
    launch_game()
    play_game()
    exit_game()

if __name__ == '__main__':
    main()
