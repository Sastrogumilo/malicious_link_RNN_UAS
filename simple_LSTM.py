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
from validasi import Cross_Validasi
from save_load_layer import Layer_Utama

#DATA SET 
dataset = "dataset.csv"   


#Start
    
#Load Dataset
df = Data(dataset).load_data()
Initial = Initial(df, df)

#print(df.sample(n=25).head(n=25))

#konversi string URL ke dalam list dimana karakter yang berisikan "printable"
#yang akan disimpan kedalam list kemudian di encode menjadi interger
url_len = Initial.konversi()
#print(url_len)

#potong URL sesuai dengan panjang maksimal array atau tambahi 0 bila url terlalu pendek #bisa disesuaikan
max_length = 75
X = Initial.potong(max_length)
print("================ Potong =====================")
print(X)

#Ekstraksi label dari form dataset ke numpy array
target = Initial.ekstark()
print("================= Ekstraksi =================")
print(target)

#check Hasil Matriks
print('Dimensi Matriks X : ', X.shape, 'Vektor Dimensi Target', target.shape)

#Cross Validasi
CV = Cross_Validasi(X, target)
CV.value()

print(CV.X_test,"\n========================\n",CV.X_train)

#Arsitektur Simple LSTM

def simple_lstm(max_len=75, emb_dim=32, max_vocab_len=100, lstm_output_size=32, W_reg=regularizers.l2(1e-4)):
    # Input
    main_input = Input(shape=(max_len,), dtype='int32', name='main_input')
    # Embedding layer
    emb = Embedding(input_dim=max_vocab_len, output_dim=emb_dim, input_length=max_len,
                dropout=0.2, W_regularizer=W_reg)(main_input) 

    # LSTM layer
    lstm = LSTM(lstm_output_size)(emb)
    lstm = Dropout(0.5)(lstm)
    
    # Output layer (last fully connected layer)
    output = Dense(1, activation='sigmoid', name='output')(lstm)

    # Compile model and define optimizer
    model = Model(input=[main_input], output=[output])
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    return model


epochs = 1
batch_size = 64

model = simple_lstm()
model.fit(CV.X_train, CV.target_train, epochs=epochs, batch_size=batch_size)
loss, accuracy = model.evaluate(CV.X_test, CV.target_test, verbose=1)

print(accuracy, "\n")
print_layer = Layer_Utama(model).print_layer_dim()