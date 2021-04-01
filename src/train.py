import sys
import glob
import time
import pickle
import numpy as np
import pandas as pd
import cv2 as cv
import mediapipe as mp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from lightgbm import LGBMClassifier

train_folder = '../data/train'
test_folder = '../data/test'
train_list = glob.glob(f'{train_folder}/*.csv')
test_list = glob.glob(f'{test_folder}/*.csv')
train_list.sort()
test_list.sort()

df_train = []
for file in train_list:
    df_action = pd.read_csv(file)
    df_train.append(df_action)
df_train = pd.concat(df_train, axis=0, ignore_index=True)

df_test = []
for file in test_list:
    df_action = pd.read_csv(file)
    df_test.append(df_action)
df_test = pd.concat(df_test, axis=0, ignore_index=True)

print(df_train['sign'].value_counts())
print(df_test['sign'].value_counts())
time.sleep(2)

# Prepare data
y_train = df_train.pop('sign')
x_train = df_train.values
y_test = df_test.pop('sign')
x_test = df_test.values

# Model
print('Training random forest')
model = RandomForestClassifier(n_estimators=9)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy of Random forest: ', accuracy)

# Save model
with open('../model/md.pickle', 'wb') as f:
    pickle.dump(model, f)
