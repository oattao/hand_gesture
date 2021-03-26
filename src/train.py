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

from config import ACTIONS

df = []
for action_name in ACTIONS:
    df_action = pd.read_csv(f'../data/{action_name}.csv')
    df.append(df_action)

df = pd.concat(df, axis=0, ignore_index=True)

df_train, df_val = train_test_split(df, test_size=0.2)
print(df_train['sign'].value_counts())
print(df_val['sign'].value_counts())
time.sleep(2)

# Prepare data
y_train = df_train.pop('sign')
x_train = df_train.values
y_val = df_val.pop('sign')
x_val = df_val.values

# Model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Test
y_pred = model.predict(x_val)
accuracy = accuracy_score(y_val, y_pred)
print('Accuracy: ', accuracy)

# Save model
with open('../model/md.pickle', 'wb') as f:
    pickle.dump(model, f)
