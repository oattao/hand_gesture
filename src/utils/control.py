import time
import pyautogui as pg

def jump():
    pg.press('space')

def liedown():
    pg.press('down')

def pause():
    pg.press('f11')
    time.sleep(0.001)
    pg.press('f11')

def switch():
    """
    Switch between hand mode and key mode
    """
    