from string import printable
import numpy as np
from keras.preprocessing import sequence

class Initial:


    def __init__(self, df, data):
        self.df = df
        self.data = data
    
        
    
    def konversi(self):
       self.df = [[printable.index(x) + \
            1 for x in url \
                if x in printable] \
                    for url in self.df.url]
               
       return self.df
    
    def potong(self, maxlen):
        return sequence.pad_sequences(self.df, maxlen = maxlen)

    
    def ekstark(self):
        return np.array(self.data.isMalicious)