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


class load_1D_LSTM:
            
        def load_model(fileModelJSON, fileWeights):
            with open(fileModelJSON, 'r') as f:
                model_json = json.load(f)
                model = model_from_json(model_json)
                
            model.load_weights(fileWeights)
            return model
                