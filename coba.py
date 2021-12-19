import pandas as pd
import numpy as np
import re, os
from string import printable
from sklearn import model_selection

#Gesim 
import tensorflow as tf 
from keras.models import Sequential, Model, model_from_json, load_model
from keras import regularizers
from keras.layers.core import Dense, Dropout, Activation, Lambda, Flatten
from keras.layers import Input, ELU, LSTM, Embedding, Convolution2D, MaxPooling2D, \
BatchNormalization, Convolution1D, MaxPooling1D, concatenate
from keras.preprocessing import sequence
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras import backend as K

from pathlib import Path
import json

import warnings
warnings.filterwarnings("ignore")

#LOCAL LIBRARY

from loadData import Data
from inisialiasasi import Initial





#Start
dataset = "dataset.csv"
df = Data.load_data(dataset)
#print(df.sample(n=25).head(n=25))

#konversi string URL ke dalam list dimana karakter yang berisikan "printable"
#yang akan disimpan kedalam list kemudian di encode menjadi interger
url = Initial.konversi(df)
#print(url)

#poton URL sesuai dengan panjang maksimal array atau tambahi 0 bila url terlalu pendek
max_len = 75 #bisa disesuaikan
X = Initial.potong(max_len, url)
#print(X)

#Ekstraksi label dari form dataset ke numpy array
target = Initial.ekstark(df)
#print(target)

#check Hasil Matriks
print('Dimensi Matriks X : ', X.shape, 'Vektor Dimensi Target', target.shape)


