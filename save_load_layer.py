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

class Layer_Utama():

    def __init__(self, model):
        self.model = model


    def print_layer_dim(self):
        l_layer = self.model.layers

        for i in range(len(l_layer)):
            print(l_layer[i])
            print("Input Layer : ", l_layer[i].input_shape, \
                "Output Layer : ", l_layer[i].output_shape)
        return True
    
    def save_model(self, fileModelJSON, fileWeights):
        if Path(fileModelJSON).is_file():
            os.remove(fileModelJSON)
        json_string = self.model.to_json()
        with open(fileModelJSON,'w' ) as f:
            json.dump(json_string, f)
        if Path(fileWeights).is_file():
            os.remove(fileWeights)
        self.model.save_weights(fileWeights)
    
    def load_model(self, fileModelJSON, fileWeights):
        with open(fileModelJSON, 'r') as f:
            model_json = json.load(f)
            self.model = model_from_json(model_json)
            
        self.model.load_weights(fileWeights)
        return self.model
       
