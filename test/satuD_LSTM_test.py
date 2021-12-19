import os, re, sys, inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#print(current_dir)
parent_dir = os.path.dirname(current_dir)
#print(parent_dir)
sys.path.insert(0, parent_dir) 

from string import printable

import tensorflow as tf
from keras.preprocessing import sequence

import satuD_LSTM_load as load_1D


#Define DATA_HOME
DATA_HOME = "../data/"

#Difine Nama Model
nama_model = "deeplearning_1DConvLSTM"

model = load_1D.load_1D_LSTM.load_model(DATA_HOME + nama_model + ".json", DATA_HOME + nama_model + ".h5")

print("Keterangan: \n \n Untuk Probabilitas diatas 50 % ("">"") adalah Malicious Link\n")
url = input("\n Masukan URL Disini > ")

url_int_tokens = [[printable.index(x) + 1 for x in url if x in printable]]
print("url_int_tokens = ", url_int_tokens)

max_len=75
X = sequence.pad_sequences(url_int_tokens, maxlen=max_len)

target_proba = model.predict(X, batch_size=1)
def print_result(proba):
    if proba > 0.5:
        return "Mencurigakan"
    else:
        return "Normal"
    
    
print("\nHasil Tes URL:", url, "adalah URL ", print_result(target_proba[0]))
print("Probabilitas :", (target_proba[0][0]*100),"%", "adalah Malicious Link")