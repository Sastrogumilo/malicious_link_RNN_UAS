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

dataset = "dataset.csv"   
#Start
    

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


## Deep Learning (1D Convolutions dan Fully Connected Layers)

def conv_fully(max_length=max_length, emb_dim=32, max_vocab=100, W_reg=regularizers.l2(1e-4)):
    main_input = Input(shape=(max_length,), dtype='int32', name='main_input')

    emb = Embedding(input_dim=max_vocab, output_dim=emb_dim, input_length=max_length,
        W_regularizer=W_reg)(main_input)

    emb = Dropout(0.25)(emb)

    def sum_1d(X):
        return K.sum(X, axis=1)
    
    def get_conv_layer(emb, kernel_size=5, filters=256):
        conv = Convolution1D(kernel_size = kernel_size, filters = filters, \
            border_mode='same')(emb)
        
        conv = ELU()(conv)

        conv = Lambda(sum_1d, output_shape=(filters,))(conv)

        conv = Dropout(0.25)(conv)

        return conv

    conv1 = get_conv_layer(emb, kernel_size=2, filters=256)
    conv2 = get_conv_layer(emb, kernel_size=3, filters=256)
    conv3 = get_conv_layer(emb, kernel_size=4, filters=256)
    conv4 = get_conv_layer(emb, kernel_size=5, filters=256)

    merge = concatenate([conv1, conv2, conv3, conv4], axis=1)

    hidden1 = Dense(1024)(merge)
    hidden1 = ELU()(hidden1)
    hidden1 = BatchNormalization(mode=0)(hidden1)
    hidden1 = Dropout(0.25)(hidden1)

    hidden2 = Dense(1024)(hidden1)
    hidden2 = ELU()(hidden2)
    hidden2 = BatchNormalization(mode=0)(hidden2)
    hidden2 = Dropout(0.5)(hidden2)

    output = Dense(1, activation='sigmoid', name='output')(hidden2)

    lr = 1e-4
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-08
    decay = 0.1


    model = Model(input=[main_input], output=[output])
    adam = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, decay=decay)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    return model

epochs = 1
batch_size = 64

model = conv_fully()
model.fit(CV.X_train, CV.target_train, epochs=epochs, batch_size=batch_size)
loss, accuracy = model.evaluate(CV.X_test, CV.target_test, verbose=1)

print(accuracy, "\n")
print_layer = Layer_Utama(model).print_layer_dim()