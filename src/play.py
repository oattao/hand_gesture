import os
import time
import webbrowser
import cv2 as cv

import pyautogui as pg

from utils.control import jump

W, H = pg.size()
EXIT_BUTTON = [2526, 20]

x_m, y_m = W//2, H//2

def launch_game():
    URL = 'chrome:dino'
    chromepath= 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s'
    controler = webbrowser.get(chromepath)
    controler.open(URL, new=1, autoraise=True)
def resume_game():
    pg.alert(text='Game resumed.', title='', button='OK')
    pg.click(x=x_m, y=y_m, button='right')

def pause_game():
    pg.click(x=-100, y=y_m, button='right')

def exit_game():
    pg.click(x=EXIT_BUTTON[0], y=EXIT_BUTTON[1], button='right')


def main():
    resume_game()

    # start game


    for i in range(5):
        pg.press('space')
        time.sleep(0.5)
        print(f'pressed {i}')    

    # after 20 click, pause game 2 seconds, then close game
    pause_game()
    time.sleep(2)
    exit_game()

    


    # for char in URL:
    #     pg.press(char)
    # pg.press('enter')
    # pg.typewrite(URL)


if __name__ == '__main__':
    main()

