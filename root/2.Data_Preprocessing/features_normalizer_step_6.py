import pandas as pd
import numpy as np
import math
from math import radians, sin, cos, acos
from sklearn import preprocessing
import os
import tqdm

def latlng(lat, lng):
    x = math.cos(lat) * math.cos(lng)
    y = math.cos(lat) * math.sin(lng)
    return (x, y)


def normalize(filename):
    df = pd.read_csv(filename)
    fname = os.path.basename(filename)

    #FILL MISSING VALUES OF LAT LNG
    df = df.fillna(df.mean())

    #COMMENTED ONLY WHEN WORKING WITH REGIONS TABLE 
    for k in (df.index):
        if k% 500 == 0:
            print("done "+ str(k)) 
        lat = df['Lat'][k]
        lng = df['Lng'][k]
        (df['Lat'][k], df['Lng'][k]) = latlng(lat, lng) 

    df.iloc[:, 1:] = preprocessing.scale(df.iloc[:, 1:])
    df.to_csv('Data Tables Step 6 - Attributes_/' + fname)



""" normalize('Data Tables Step 5 - Attributes\school.csv')
print("Stage complete")
normalize('Data Tables Step 5 - Attributes\\train.csv')
print("Stage complete")

normalize('Data Tables Step 5 - Attributes\property.csv')
print("Stage complete") """
normalize('Data Tables Step 5 - Attributes\\region_modified.csv')
print("Stage complete")
